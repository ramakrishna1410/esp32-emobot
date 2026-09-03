// Main board firmware entry point (ESP32-S3, round display).
//
// This is a scaffold only — see docs/roadmap.md for what Phase 1 needs to
// add here: face-renderer state machine, wake-word + VAD, the cloud-LLM /
// scripted-fallback response engine, BME280 climate reads, and the BLE
// life-event listener.

#include <Arduino.h>

#if defined(BOT_ROLE_PARENT)
static const char *kBotRole = "parent-bot";
#elif defined(BOT_ROLE_KID)
static const char *kBotRole = "kid-bot";
#else
#error "Define BOT_ROLE_PARENT or BOT_ROLE_KID via the PlatformIO environment"
#endif

void setup() {
  Serial.begin(115200);
  Serial.printf("emobot main-board booting, role=%s\n", kBotRole);

  // TODO(Phase 1): face renderer init
  // TODO(Phase 1): mic + wake-word (ESP-SR) init
  // TODO(Phase 1): BME280 init
  // TODO(Phase 1): BLE life-event listener init
  // TODO(Phase 1B): UART link to ESP32-CAM co-processor
  // TODO(Phase 2): LoRa link init
}

void loop() {
  // TODO: drive the face-renderer state machine
  // TODO: poll wake-word engine, dispatch to response engine on trigger
  delay(10);
}
