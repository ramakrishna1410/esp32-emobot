// Public API for the on-device llama2.c-style inference engine.
//
// Ported from vendor/llama2c/runq.c (int8-quantized inference). The neural
// net math (rmsnorm/softmax/matmul/forward/encode/sample) is copied
// essentially unchanged — it's portable C with no platform dependencies.
// What's ESP32-specific and different from upstream:
//   - Model weights are read via esp_partition_mmap() (zero-copy, direct
//     flash read) from the "model" raw data partition instead of fopen+mmap
//     on a file — see partitions.csv.
//   - The tokenizer is read via esp_partition_read() from the "tokenizer"
//     partition into a small RAM buffer (it's a few KB, no need for mmap).
//   - RunState (KV cache, activation buffers) is allocated in PSRAM via
//     heap_caps_calloc(..., MALLOC_CAP_SPIRAM) instead of plain calloc().
//   - There is no CLI/chat mode; generation streams through a callback
//     instead of printf, so the caller (main.cpp) decides where output goes
//     (Serial now, a TTS module later).
//
// NOT YET COMPILE-TESTED on real hardware in this repo (no ESP32 toolchain
// / network access to PlatformIO's registry in the dev sandbox that wrote
// this). Build and flash it yourself, then fix forward from any compiler
// or runtime errors — see firmware/main-board/README.md.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Initializes the transformer + tokenizer + sampler from the "model" and
// "tokenizer" flash partitions. Call once at boot. Returns 0 on success,
// nonzero on failure (partition not found, bad magic/version, OOM, ...).
int llm_init(void);

// Callback invoked once per generated text piece (a decoded token's string
// fragment — not necessarily a whole word). Concatenate pieces to build the
// full response.
typedef void (*llm_piece_cb)(const char *piece);

// Generates a response for `prompt` (already formatted as
// "User: <trigger>\nBot:" per the training format — see
// llm-training/README.md's "literal \n marker" note, and pass the SAME
// literal two-character sequence here, not an actual newline).
// Streams each output piece to `on_piece` as it's generated, stops at the
// model's BOS token (same end-of-sequence convention as upstream run.c) or
// after `max_tokens`, whichever comes first.
// `temperature` 0.0 = greedy/deterministic (recommended for now — see
// llm-training/README.md's note on sampled decoding sometimes retrieving
// the wrong category at the current dataset size). >0.0 enables
// temperature + top-p sampling (topp fixed at 0.9, matching upstream's
// default).
void llm_generate(const char *prompt, int max_tokens, float temperature,
                   llm_piece_cb on_piece);

#ifdef __cplusplus
}
#endif
