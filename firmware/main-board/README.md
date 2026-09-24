# Main board firmware — ESP32-S3 N16R8

On-device llama2.c-style inference, ported from `llm-training/vendor/llama2c/runq.c`.
See `include/llm_infer.h` and `src/llm_infer.c` for what changed from upstream
and why — short version: weights are read via zero-copy flash mmap instead
of a file, RunState/KV-cache lives in PSRAM, and generation streams through
a callback instead of printf.

## Status: written, not yet compile-tested or flashed

This code was written in a sandboxed dev environment with **no network
access to PlatformIO's platform registry and no physical ESP32 board** — so
it has not been compiled or run. It's a careful, real port (not a stub),
but treat the first build as a debugging session, not a sure thing. Report
back any compiler/linker errors or runtime crashes so they can be fixed.

## Build + flash steps

1. **Build and upload the firmware app** (this does NOT include the model —
   that's a separate step below):
   ```
   cd firmware/main-board
   pio run -t upload -t monitor
   ```
   This flashes `partitions.csv`'s partition table along with the compiled
   app. On first boot, `llm_init()` will fail (no model flashed yet) —
   that's expected until step 2.

2. **Train/export a model** (if you haven't already — see
   `llm-training/README.md`), producing e.g.
   `llm-training/out/conv_v3/model_q8.bin` and the matching tokenizer, e.g.
   `llm-training/data/tok512.bin`.

3. **Flash the model + tokenizer** to their raw data partitions:
   ```
   tools/flash_model.sh /dev/ttyUSB0 \
       llm-training/out/conv_v3/model_q8.bin \
       llm-training/data/tok512.bin
   ```
   (Find your board's serial port first — `pio device list`.)

4. **Reset the board** and open a serial monitor (`pio device monitor`, or
   the `-t monitor` from step 1). You should see `llm_init()` succeed and a
   prompt to type a test line, e.g.:
   ```
   User: I feel sad today.\nBot:
   ```
   (type a literal backslash-n between User/Bot, not an actual newline —
   see llm-training/README.md's note on why the training format uses this
   marker).

## Known unknowns to verify on real hardware

- **`board_build.arduino.memory_type = qio_opi`** in `platformio.ini` is a
  guess at the correct octal-PSRAM mode for N16R8-class modules — verify
  against your exact board's datasheet; wrong PSRAM mode is a common cause
  of boot loops/garbage on ESP32-S3 boards with octal PSRAM.
- **`esp_partition_mmap()` region size**: the "model" partition is 2MB;
  ESP32-S3's flash mmap address space has finite capacity shared with the
  running app's own code mapping. This should be fine at 2MB but hasn't
  been confirmed — if `esp_partition_mmap` fails, check the IDF log for the
  specific error and search "ESP32-S3 esp_partition_mmap size limit".
- **PSRAM allocation sizing**: `malloc_run_state()` computes KV-cache size
  as `n_layers * seq_len * kv_dim * 4 bytes * 2` (key+value). For the
  current smoke-test config (`dim=64, n_layers=5, seq_len=512` — note the
  *architecture's* max seq_len, not the shorter `max_seq_len` used during
  the CPU smoke-test training runs) this is small; re-check against the 8MB
  PSRAM budget if you train a larger model.
- **Tokens/sec on real hardware** is unmeasured — the `achieved tok/s` log
  line will report it once this actually boots; compare against the CPU
  numbers in `llm-training/README.md` as a sanity check (expect this to be
  slower than a modern laptop's CPU, faster than nothing).
