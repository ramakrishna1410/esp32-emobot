// Main board firmware entry point (ESP32-S3 N16R8, round display).
//
// Phase 2 hardware-proof milestone: no display/mic/TTS integration yet
// (that's Phase 5) — this just proves the on-device LLM inference pipeline
// actually runs on the real chip. Type a line over Serial in the training
// format ("User: <trigger>\nBot:" — see llm-training/README.md's literal
// \n-marker note) and it prints the generated response back.
//
// NOT YET FLASHED/TESTED on real hardware — see README.md in this
// directory for the build+flash steps to run yourself.

#include <Arduino.h>
#include "llm_infer.h"

static void on_piece(const char *piece) {
  Serial.print(piece);
}

void setup() {
  Serial.begin(115200);
  delay(1000);  // give the serial monitor time to attach
  Serial.println("emobot main-board booting");

  Serial.println("Initializing on-device LLM (this reads the model+tokenizer flash partitions)...");
  int err = llm_init();
  if (err != 0) {
    Serial.printf("llm_init() failed with code %d — check the Serial log above for the ESP_LOGE detail,\n", err);
    Serial.println("and confirm the model/tokenizer partitions were flashed (see tools/flash_model.sh).");
    while (1) delay(1000);
  }
  Serial.println("LLM ready. Type a prompt ending in 'Bot:' and press enter, e.g.:");
  Serial.println("  User: I feel sad today.\\nBot:");
  Serial.println("(note: type a literal backslash-n, not an actual newline, between User/Bot — matches training format)");
}

void loop() {
  static String line;
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      if (line.length() > 0) {
        Serial.printf("\n--- generating (greedy, deterministic) ---\n");
        unsigned long start_ms = millis();
        llm_generate(line.c_str(), /*max_tokens=*/60, /*temperature=*/0.0f, on_piece);
        Serial.printf("\n--- done, %lu ms wall clock ---\n", millis() - start_ms);
        line = "";
      }
    } else if (c != '\r') {
      line += c;
    }
  }
}
