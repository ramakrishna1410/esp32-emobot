// ESP32-CAM vision co-processor firmware entry point (Phase 1B).
//
// Scaffold only. Per docs/roadmap.md Phase 1B: bring up ESP-WHO face
// detection standalone first, confirm frame rate/reliability, before adding
// an emotion classifier on top. Sends a compact result to the main board
// over UART, e.g. `{face_detected, emotion, confidence}`.

#include <Arduino.h>

void setup() {
  Serial.begin(115200);
  Serial.println("emobot cam-coproc booting");

  // TODO(Phase 1B): camera init (ESP-WHO / esp32-camera driver)
  // TODO(Phase 1B): face detection pipeline
  // TODO(Phase 1B): UART link to main board
}

void loop() {
  // TODO: run detection loop, send result over UART on change
  delay(10);
}
