// On-device llama2.c-style int8 inference for ESP32-S3.
//
// Ported from vendor/llama2c/runq.c — see include/llm_infer.h for what
// changed and why. The neural net math below (rmsnorm/softmax/matmul/
// forward/encode/sample) is copied essentially unchanged from upstream:
// it's portable C, no platform dependencies. Section headers match
// upstream's so the two files are easy to diff.
//
// NOT YET COMPILE-TESTED on real hardware — see llm_infer.h and
// firmware/main-board/README.md.

#include "llm_infer.h"

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "esp_partition.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "esp_log.h"

static const char *TAG = "llm_infer";

// ----------------------------------------------------------------------------
// Globals (matches upstream: GS is a global set once from the checkpoint header)
static int GS = 0;

// ----------------------------------------------------------------------------
// Transformer model structs — identical to upstream except Transformer's
// bookkeeping fields (fd/data/file_size for POSIX mmap) are replaced with
// an ESP-IDF partition mmap handle.

typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
} Config;

typedef struct {
    int8_t *q;
    float *s;
} QuantizedTensor;

typedef struct {
    QuantizedTensor *q_tokens;
    float *token_embedding_table;
    float *rms_att_weight;
    float *rms_ffn_weight;
    QuantizedTensor *wq;
    QuantizedTensor *wk;
    QuantizedTensor *wv;
    QuantizedTensor *wo;
    QuantizedTensor *w1;
    QuantizedTensor *w2;
    QuantizedTensor *w3;
    float *rms_final_weight;
    QuantizedTensor *wcls;
} TransformerWeights;

typedef struct {
    float *x;
    float *xb;
    float *xb2;
    float *hb;
    float *hb2;
    QuantizedTensor xq;
    QuantizedTensor hq;
    float *q;
    float *k;
    float *v;
    float *att;
    float *logits;
    float *key_cache;
    float *value_cache;
} RunState;

typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
    esp_partition_mmap_handle_t mmap_handle;  // ESP32: for esp_partition_munmap cleanup (unused — we never tear down)
} Transformer;

static Transformer g_transformer;
static int g_initialized = 0;

// ----------------------------------------------------------------------------
// RunState allocation — ESP32 change: allocate in PSRAM (heap_caps_calloc
// with MALLOC_CAP_SPIRAM) instead of plain calloc(). The KV cache in
// particular (n_layers * seq_len * kv_dim floats, x2 for K and V) is the
// biggest consumer and needs PSRAM's 8MB, not the ~512KB of internal SRAM.

static int malloc_run_state(RunState *s, Config *p) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = heap_caps_calloc(p->dim, sizeof(float), MALLOC_CAP_SPIRAM);
    s->xb = heap_caps_calloc(p->dim, sizeof(float), MALLOC_CAP_SPIRAM);
    s->xb2 = heap_caps_calloc(p->dim, sizeof(float), MALLOC_CAP_SPIRAM);
    s->hb = heap_caps_calloc(p->hidden_dim, sizeof(float), MALLOC_CAP_SPIRAM);
    s->hb2 = heap_caps_calloc(p->hidden_dim, sizeof(float), MALLOC_CAP_SPIRAM);
    s->xq.q = heap_caps_calloc(p->dim, sizeof(int8_t), MALLOC_CAP_SPIRAM);
    s->xq.s = heap_caps_calloc(p->dim, sizeof(float), MALLOC_CAP_SPIRAM);
    s->hq.q = heap_caps_calloc(p->hidden_dim, sizeof(int8_t), MALLOC_CAP_SPIRAM);
    s->hq.s = heap_caps_calloc(p->hidden_dim, sizeof(float), MALLOC_CAP_SPIRAM);
    s->q = heap_caps_calloc(p->dim, sizeof(float), MALLOC_CAP_SPIRAM);
    s->k = heap_caps_calloc(kv_dim, sizeof(float), MALLOC_CAP_SPIRAM);
    s->v = heap_caps_calloc(kv_dim, sizeof(float), MALLOC_CAP_SPIRAM);
    s->att = heap_caps_calloc(p->n_heads * p->seq_len, sizeof(float), MALLOC_CAP_SPIRAM);
    s->logits = heap_caps_calloc(p->vocab_size, sizeof(float), MALLOC_CAP_SPIRAM);
    s->key_cache = heap_caps_calloc((size_t)p->n_layers * p->seq_len * kv_dim, sizeof(float), MALLOC_CAP_SPIRAM);
    s->value_cache = heap_caps_calloc((size_t)p->n_layers * p->seq_len * kv_dim, sizeof(float), MALLOC_CAP_SPIRAM);
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->xq.q || !s->xq.s ||
        !s->hq.q || !s->hq.s || !s->q || !s->k || !s->v || !s->att || !s->logits ||
        !s->key_cache || !s->value_cache) {
        ESP_LOGE(TAG, "PSRAM allocation failed for RunState — model/seq_len too big for available PSRAM");
        return -1;
    }
    return 0;
}

// ----------------------------------------------------------------------------
// Quantization functions — unchanged from upstream.

static void dequantize(QuantizedTensor *qx, float *x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = qx->q[i] * qx->s[i / GS];
    }
}

static void quantize(QuantizedTensor *qx, float *x, int n) {
    int num_groups = n / GS;
    float Q_MAX = 127.0f;
    for (int group = 0; group < num_groups; group++) {
        float wmax = 0.0;
        for (int i = 0; i < GS; i++) {
            float val = fabsf(x[group * GS + i]);
            if (val > wmax) wmax = val;
        }
        float scale = wmax / Q_MAX;
        qx->s[group] = scale;
        for (int i = 0; i < GS; i++) {
            float quant_value = x[group * GS + i] / scale;
            int8_t quantized = (int8_t)roundf(quant_value);
            qx->q[group * GS + i] = quantized;
        }
    }
}

static QuantizedTensor *init_quantized_tensors(void **ptr, int n, int size_each) {
    void *p = *ptr;
    // NOTE: this struct array itself is small (n * 8 bytes) — fine on internal heap.
    QuantizedTensor *res = malloc(n * sizeof(QuantizedTensor));
    for (int i = 0; i < n; i++) {
        res[i].q = (int8_t *)p;
        p = (int8_t *)p + size_each;
        res[i].s = (float *)p;
        p = (float *)p + size_each / GS;
    }
    *ptr = p;
    return res;
}

static void memory_map_weights(TransformerWeights *w, Config *p, void *ptr, uint8_t shared_classifier) {
    int head_size = p->dim / p->n_heads;
    float *fptr = (float *)ptr;
    w->rms_att_weight = fptr;
    fptr += p->n_layers * p->dim;
    w->rms_ffn_weight = fptr;
    fptr += p->n_layers * p->dim;
    w->rms_final_weight = fptr;
    fptr += p->dim;

    ptr = (void *)fptr;
    w->q_tokens = init_quantized_tensors(&ptr, 1, p->vocab_size * p->dim);
    // Dequantized token embedding table: vocab_size * dim floats. For our
    // tiny models (e.g. 512 * 64 * 4 bytes = 128KB) this is fine in PSRAM;
    // re-check this against PSRAM budget if vocab_size/dim grow a lot.
    w->token_embedding_table = heap_caps_malloc((size_t)p->vocab_size * p->dim * sizeof(float), MALLOC_CAP_SPIRAM);
    dequantize(w->q_tokens, w->token_embedding_table, p->vocab_size * p->dim);

    w->wq = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_heads * head_size));
    w->wk = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * head_size));
    w->wv = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * head_size));
    w->wo = init_quantized_tensors(&ptr, p->n_layers, (p->n_heads * head_size) * p->dim);
    w->w1 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);
    w->w2 = init_quantized_tensors(&ptr, p->n_layers, p->hidden_dim * p->dim);
    w->w3 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);
    w->wcls = shared_classifier ? w->q_tokens : init_quantized_tensors(&ptr, 1, p->dim * p->vocab_size);
}

// ----------------------------------------------------------------------------
// ESP32-specific: load the checkpoint from the "model" flash partition via
// esp_partition_mmap (zero-copy — weight pointers above point directly into
// mapped flash, same as upstream's weight pointers point into mmap'd file
// data). Replaces upstream's read_checkpoint()+build_transformer().

static int build_transformer_esp32(Transformer *t) {
    const esp_partition_t *part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "model");
    if (!part) {
        ESP_LOGE(TAG, "'model' partition not found — check partitions.csv is flashed and the model was written to it");
        return -1;
    }

    const void *mapped = NULL;
    esp_err_t err = esp_partition_mmap(part, 0, part->size, ESP_PARTITION_MMAP_DATA, &mapped, &t->mmap_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_partition_mmap('model') failed: %s", esp_err_to_name(err));
        return -2;
    }

    const uint8_t *base = (const uint8_t *)mapped;
    size_t off = 0;
    uint32_t magic_number;
    memcpy(&magic_number, base + off, sizeof(magic_number));
    off += sizeof(magic_number);
    if (magic_number != 0x616b3432) {
        ESP_LOGE(TAG, "bad magic number in model partition (expected llama2.c 'ak42' header) — did you flash the right file?");
        return -3;
    }
    int version;
    memcpy(&version, base + off, sizeof(version));
    off += sizeof(version);
    if (version != 2) {
        ESP_LOGE(TAG, "bad checkpoint version %d, need version 2 (export.py --version 2)", version);
        return -4;
    }
    memcpy(&t->config, base + off, sizeof(Config));
    off += sizeof(Config);
    uint8_t shared_classifier;
    memcpy(&shared_classifier, base + off, sizeof(shared_classifier));
    off += sizeof(shared_classifier);
    int group_size;
    memcpy(&group_size, base + off, sizeof(group_size));
    off += sizeof(group_size);
    GS = group_size;

    const int header_size = 256;  // fixed by llama2.c's version-2 export format
    void *weights_ptr = (void *)(base + header_size);
    memory_map_weights(&t->weights, &t->config, weights_ptr, shared_classifier);

    ESP_LOGI(TAG, "model loaded: dim=%d n_layers=%d n_heads=%d vocab_size=%d seq_len=%d (GS=%d)",
             t->config.dim, t->config.n_layers, t->config.n_heads, t->config.vocab_size, t->config.seq_len, GS);

    return malloc_run_state(&t->state, &t->config);
}

// ----------------------------------------------------------------------------
// neural net blocks — unchanged from upstream (portable float math).
// Note: upstream's "#pragma omp parallel for" on matmul()/the attention
// head loop is a no-op here (this build has no OpenMP) — runs single-
// threaded on one core. Pinning the second core to help is a possible
// later optimization, not needed for the Phase 2 hardware-proof milestone.

static void rmsnorm(float *o, float *x, float *weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) ss += x[j] * x[j];
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) o[j] = weight[j] * (ss * x[j]);
}

static void softmax(float *x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) x[i] /= sum;
}

static void matmul(float *xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        int32_t ival = 0;
        int in = i * n;
        int j;
        for (j = 0; j <= n - GS; j += GS) {
            for (int k = 0; k < GS; k++) {
                ival += ((int32_t)x->q[j + k]) * ((int32_t)w->q[in + j + k]);
            }
            val += ((float)ival) * w->s[(in + j) / GS] * x->s[j / GS];
            ival = 0;
        }
        xout[i] = val;
    }
}

static float *forward(Transformer *transformer, int token, int pos) {
    Config *p = &transformer->config;
    TransformerWeights *w = &transformer->weights;
    RunState *s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    memcpy(x, w->token_embedding_table + token * dim, dim * sizeof(float));

    for (int l = 0; l < p->n_layers; l++) {
        rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);

        quantize(&s->xq, s->xb, dim);
        matmul(s->q, &s->xq, w->wq + l, dim, dim);
        matmul(s->k, &s->xq, w->wk + l, dim, kv_dim);
        matmul(s->v, &s->xq, w->wv + l, dim, kv_dim);

        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1;
            for (int v = 0; v < rotn; v++) {
                float *vec = v == 0 ? s->q : s->k;
                float v0 = vec[i];
                float v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        int loff = l * p->seq_len * kv_dim;
        float *key_cache_row = s->key_cache + loff + pos * kv_dim;
        float *value_cache_row = s->value_cache + loff + pos * kv_dim;
        memcpy(key_cache_row, s->k, kv_dim * sizeof(*key_cache_row));
        memcpy(value_cache_row, s->v, kv_dim * sizeof(*value_cache_row));

        for (int h = 0; h < p->n_heads; h++) {
            float *q = s->q + h * head_size;
            float *att = s->att + h * p->seq_len;
            for (int t = 0; t <= pos; t++) {
                float *k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) score += q[i] * k[i];
                score /= sqrtf(head_size);
                att[t] = score;
            }
            softmax(att, pos + 1);
            float *xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float *v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att[t];
                for (int i = 0; i < head_size; i++) xb[i] += a * v[i];
            }
        }

        quantize(&s->xq, s->xb, dim);
        matmul(s->xb2, &s->xq, w->wo + l, dim, dim);
        for (int i = 0; i < dim; i++) x[i] += s->xb2[i];

        rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);
        quantize(&s->xq, s->xb, dim);
        matmul(s->hb, &s->xq, w->w1 + l, dim, hidden_dim);
        matmul(s->hb2, &s->xq, w->w3 + l, dim, hidden_dim);
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + expf(-val)));
            val *= s->hb2[i];
            s->hb[i] = val;
        }
        quantize(&s->hq, s->hb, hidden_dim);
        matmul(s->xb, &s->hq, w->w2 + l, hidden_dim, dim);
        for (int i = 0; i < dim; i++) x[i] += s->xb[i];
    }

    rmsnorm(x, x, w->rms_final_weight, dim);
    quantize(&s->xq, x, dim);
    matmul(s->logits, &s->xq, w->wcls, dim, p->vocab_size);
    return s->logits;
}

// ----------------------------------------------------------------------------
// Tokenizer — same struct/algorithm as upstream. ESP32 change: build_tokenizer
// reads the whole "tokenizer" partition into one PSRAM buffer up front (it's
// small — a few KB to low tens of KB) and parses sequentially from that
// buffer instead of fread()ing a file, via the tok_read() helper below.

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char **vocab;
    float *vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
} Tokenizer;

static Tokenizer g_tokenizer;

static const uint8_t *tok_buf;
static size_t tok_off;
static void tok_read(void *dst, size_t n) {
    memcpy(dst, tok_buf + tok_off, n);
    tok_off += n;
}

static int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str);
}

static int build_tokenizer_esp32(Tokenizer *t, int vocab_size) {
    const esp_partition_t *part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "tokenizer");
    if (!part) {
        ESP_LOGE(TAG, "'tokenizer' partition not found — check partitions.csv is flashed and tokenizer.bin was written to it");
        return -1;
    }
    uint8_t *buf = heap_caps_malloc(part->size, MALLOC_CAP_SPIRAM);
    if (!buf) {
        ESP_LOGE(TAG, "PSRAM allocation failed for tokenizer buffer");
        return -2;
    }
    esp_err_t err = esp_partition_read(part, 0, buf, part->size);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_partition_read('tokenizer') failed: %s", esp_err_to_name(err));
        return -3;
    }
    tok_buf = buf;
    tok_off = 0;

    t->vocab_size = vocab_size;
    t->vocab = malloc(vocab_size * sizeof(char *));
    t->vocab_scores = malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL;
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }

    tok_read(&t->max_token_length, sizeof(int));
    for (int i = 0; i < vocab_size; i++) {
        tok_read(t->vocab_scores + i, sizeof(float));
        int len;
        tok_read(&len, sizeof(int));
        t->vocab[i] = malloc(len + 1);
        tok_read(t->vocab[i], len);
        t->vocab[i][len] = '\0';
    }
    return 0;
}

static char *decode(Tokenizer *t, int prev_token, int token) {
    char *piece = t->vocab[token];
    if (prev_token == 1 && piece[0] == ' ') piece++;
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char *)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

static void safe_print_piece(char *piece, llm_piece_cb cb) {
    if (piece == NULL || piece[0] == '\0') return;
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) return;
    }
    cb(piece);
}

static int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    TokenIndex tok = {.str = str};
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

static void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    if (text == NULL) return;

    if (t->sorted_vocab == NULL) {
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    char *str_buffer = malloc((t->max_token_length * 2 + 1 + 2) * sizeof(char));
    size_t str_len = 0;
    *n_tokens = 0;

    if (bos) tokens[(*n_tokens)++] = 1;

    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    for (char *c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) str_len = 0;
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';
        if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) continue;

        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            for (size_t i = 0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0;
    }

    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;
        for (int i = 0; i < (*n_tokens - 1); i++) {
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }
        if (best_idx == -1) break;
        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++) tokens[i] = tokens[i + 1];
        (*n_tokens)--;
    }

    if (eos) tokens[(*n_tokens)++] = 2;
    free(str_buffer);
}

// ----------------------------------------------------------------------------
// Sampler — unchanged from upstream, except time_in_ms() uses esp_timer_get_time()
// instead of clock_gettime(), and the RNG is seeded from esp_timer_get_time() too
// (no time(NULL) wall clock on a device with no RTC/network time sync).

typedef struct {
    float prob;
    int index;
} ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex *probindex;
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

static int sample_argmax(float *probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

static int sample_mult(float *probabilities, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) return i;
    }
    return n - 1;
}

static int compare_probindex(const void *a, const void *b) {
    ProbIndex *a_ = (ProbIndex *)a;
    ProbIndex *b_ = (ProbIndex *)b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

static int sample_topp(float *probabilities, int n, float topp, ProbIndex *probindex, float coin) {
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare_probindex);

    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1;
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break;
        }
    }

    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) return probindex[i].index;
    }
    return probindex[last_idx].index;
}

static unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
static float random_f32(unsigned long long *state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

static int sample(Sampler *sampler, float *logits) {
    int next;
    if (sampler->temperature == 0.0f) {
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        for (int q = 0; q < sampler->vocab_size; q++) logits[q] /= sampler->temperature;
        softmax(logits, sampler->vocab_size);
        float coin = random_f32(&sampler->rng_state);
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// Public API

int llm_init(void) {
    if (g_initialized) return 0;

    memset(&g_transformer, 0, sizeof(g_transformer));
    int err = build_transformer_esp32(&g_transformer);
    if (err != 0) return err;

    err = build_tokenizer_esp32(&g_tokenizer, g_transformer.config.vocab_size);
    if (err != 0) return err;

    g_initialized = 1;
    return 0;
}

void llm_generate(const char *prompt, int max_tokens, float temperature, llm_piece_cb on_piece) {
    if (!g_initialized) {
        ESP_LOGE(TAG, "llm_generate() called before llm_init() succeeded");
        return;
    }
    if (max_tokens <= 0 || max_tokens > g_transformer.config.seq_len) {
        max_tokens = g_transformer.config.seq_len;
    }

    Sampler sampler;
    sampler.vocab_size = g_tokenizer.vocab_size;
    sampler.temperature = temperature;
    sampler.topp = 0.9f;
    sampler.rng_state = (unsigned long long)esp_timer_get_time();
    sampler.probindex = malloc(sampler.vocab_size * sizeof(ProbIndex));

    char *prompt_str = (char *)(prompt ? prompt : "");
    int num_prompt_tokens = 0;
    int *prompt_tokens = malloc((strlen(prompt_str) + 3) * sizeof(int));
    encode(&g_tokenizer, prompt_str, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        ESP_LOGE(TAG, "encode() produced 0 tokens for prompt");
        free(prompt_tokens);
        free(sampler.probindex);
        return;
    }

    int64_t start_us = 0;
    int next;
    int token = prompt_tokens[0];
    int pos = 0;
    while (pos < max_tokens) {
        float *logits = forward(&g_transformer, token, pos);

        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample(&sampler, logits);
        }
        pos++;

        if (next == 1) break;  // BOS token delimits sequences, same as upstream

        char *piece = decode(&g_tokenizer, token, next);
        safe_print_piece(piece, on_piece);
        token = next;

        if (start_us == 0) start_us = esp_timer_get_time();
    }

    if (pos > 1) {
        int64_t end_us = esp_timer_get_time();
        double tok_per_sec = (pos - 1) / ((end_us - start_us) / 1e6);
        ESP_LOGI(TAG, "achieved tok/s: %.2f", tok_per_sec);
    }

    free(prompt_tokens);
    free(sampler.probindex);
}
