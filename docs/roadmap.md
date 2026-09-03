# Roadmap

## Phase 1 — Core companion (single unit, MVP)
- Round-display face animation engine (state machine, sprite eyes).
- Wake-word (ESP-SR WakeNet) + Voice Activity Detection for hands-free
  listening; simple audio-level mood heuristic as one fallback input.
- Cloud-LLM-first response engine (wake-word → check WiFi → STT/LLM/TTS if
  up, else local scripted fallback), with a short follow-up-listening window
  after each response for natural back-and-forth.
- Room climate: BME280 (temp/humidity), feeding either the LLM prompt
  (online) or a scripted climate-alert line (offline).
- BLE life-event trigger from a phone-side shortcut (e.g. "movie booked").
- Ambient light sensor → day/night display behavior.

## Phase 1B — Camera-based emotion (adds the ESP32-CAM co-processor)
- Bring up ESP32-CAM standalone (ESP-WHO face detection) before adding
  emotion classification.
- Wire ESP32-CAM → main board over UART with a compact result message format.
- Start with face-presence + a crude heuristic before a trained classifier.
- Merge camera-based mood signal with the Phase 1 mic-based signal.

## Phase 2 — Twin bots (parent/kid pairing)
- Second identical unit as `kid-bot`.
- LoRa pairing + "call kid home" signal flow.
- Presence/proximity via LoRa RSSI/heartbeat, GPS layered in once outdoors.
- Fan-left-on / energy-waste detection (current-clamp sensor + presence
  sensor), local alert if home, optional WiFi→push alert if away.

## Phase 3 — Stretch goals
- Function-calling / tools for the LLM backend (stock-price lookup,
  room-climate query using live sensor data).
- Simpler non-LLM stock-price fallback path (cached tickers).
- Outdoor GPS distance/direction refinement.
- Improve the Phase 1B emotion classifier's accuracy/class count.
- Noted-but-not-planned: a Rockchip-class SBC (real NPU, runs Linux) could
  run a small LLM fully offline — considered and explicitly declined in favor
  of staying on the ESP32-only stack with a cloud LLM. Revisit only if the
  WiFi-dependency trade-off proves unworkable in practice.

See the original brainstorm plan for full context and reasoning behind these
choices.
