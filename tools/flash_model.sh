#!/usr/bin/env bash
# Flashes a trained llama2.c model + tokenizer to the ESP32-S3 N16R8's
# "model" and "tokenizer" raw data partitions (see
# firmware/main-board/partitions.csv for the offsets below).
#
# These are written directly with esptool.py, NOT through PlatformIO's
# normal firmware upload — the model/tokenizer partitions hold raw binary
# data read via esp_partition_mmap()/esp_partition_read(), not a filesystem
# or app image, so this is the equivalent of flashing a second, separate
# binary blob at a fixed flash offset.
#
# Usage:
#   tools/flash_model.sh <port> <model_q8.bin> <tokenizer.bin>
# Example:
#   tools/flash_model.sh /dev/ttyUSB0 \
#       llm-training/out/conv_v3/model_q8.bin \
#       llm-training/data/tok512.bin
#
# Offsets MUST match firmware/main-board/partitions.csv exactly — if you
# change that file, update these too.
set -euo pipefail

PORT="${1:?Usage: $0 <port> <model_q8.bin> <tokenizer.bin>}"
MODEL_BIN="${2:?Usage: $0 <port> <model_q8.bin> <tokenizer.bin>}"
TOKENIZER_BIN="${3:?Usage: $0 <port> <model_q8.bin> <tokenizer.bin>}"

MODEL_OFFSET=0x310000
MODEL_PARTITION_SIZE=$((0x200000))
TOKENIZER_OFFSET=0x510000
TOKENIZER_PARTITION_SIZE=$((0x10000))

model_size=$(stat -c%s "$MODEL_BIN" 2>/dev/null || stat -f%z "$MODEL_BIN")
tok_size=$(stat -c%s "$TOKENIZER_BIN" 2>/dev/null || stat -f%z "$TOKENIZER_BIN")

if [ "$model_size" -gt "$MODEL_PARTITION_SIZE" ]; then
  echo "ERROR: $MODEL_BIN is $model_size bytes, larger than the model partition ($MODEL_PARTITION_SIZE bytes)." >&2
  echo "Resize the 'model' partition in firmware/main-board/partitions.csv (and this script's offsets) first." >&2
  exit 1
fi
if [ "$tok_size" -gt "$TOKENIZER_PARTITION_SIZE" ]; then
  echo "ERROR: $TOKENIZER_BIN is $tok_size bytes, larger than the tokenizer partition ($TOKENIZER_PARTITION_SIZE bytes)." >&2
  exit 1
fi

echo "Flashing model ($model_size bytes) to $MODEL_OFFSET ..."
esptool.py --port "$PORT" write_flash "$MODEL_OFFSET" "$MODEL_BIN"

echo "Flashing tokenizer ($tok_size bytes) to $TOKENIZER_OFFSET ..."
esptool.py --port "$PORT" write_flash "$TOKENIZER_OFFSET" "$TOKENIZER_BIN"

echo "Done. Firmware (built/uploaded separately via 'pio run -t upload') will read these on next boot."
