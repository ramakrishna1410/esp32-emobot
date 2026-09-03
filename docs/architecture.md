# Architecture

## Three chips per toy

Each physical unit (`parent-bot`, `kid-bot`) combines three cooperating boards:

- **Main board (ESP32-S3, round display)** — owns the UI (face animation on the
  round LCD, touch), local audio (mic capture, speaker playback), and all
  decision-making. Everything else reports into this board.
- **ESP32-CAM (vision co-processor)** — runs face detection and a small
  quantized emotion classifier on-device (Espressif ESP-WHO/ESP-DL), sends a
  compact result to the main board over UART, e.g.:
  `{face_detected: true, emotion: "sad", confidence: 0.7}`.
- **LoRa+GPS module (comms co-processor)** — the offline long-range link
  between `parent-bot` and `kid-bot` (presence heartbeat, "come home"
  signaling), plus outdoor GPS for distance/direction once the GPS half has a
  sky-visible fix. GPS is not used for indoor localization — it doesn't work
  through walls/roofs; LoRa handles the indoor/house-to-house case instead.

## Online vs. offline modes

The core loop (face animation, mood sensing via mic/camera, room-climate
reads, BLE life-events, LoRa twin-bot signaling) works with **zero
connectivity**. WiFi is used only opportunistically:

- **Conversation**: wake-word (ESP-SR WakeNet, on-device) triggers a listen
  window. If WiFi is up, the utterance streams to a cloud backend
  (STT → LLM → TTS) for a freshly generated response — this is the primary
  conversation path per project decision. If WiFi is down, a local scripted
  response engine (pre-written lines/audio clips per trigger category,
  randomized non-repeating selection) answers instead, so the bot is never
  silent offline — just less varied.
- **Stock price / away-from-home alerts**: WiFi-gated, best-effort. Must fail
  fast/silently when no known network is present rather than blocking the
  core loop.

## Message flow: mood detection to response

1. Mic (mood heuristic) and/or ESP32-CAM (face+emotion) report a signal to the
   main board.
2. Main board checks WiFi.
   - **Online**: builds a short prompt from the signal (e.g. "the child looks
     sad, react warmly, 1-2 sentences") and sends it to the cloud LLM backend.
     Plays back the generated audio response.
   - **Offline**: looks up the trigger category in the local scripted table
     and plays a pre-written line.
3. Face renderer updates the display to match (e.g. comforting expression).

## LoRa twin-bot flow ("call kid home")

1. `parent-bot` gets a trigger (button, or a voice command directed at the
   LLM/local engine).
2. `parent-bot` sends a short message over LoRa to `kid-bot`.
3. `kid-bot` renders a "mommy wants you" expression and plays a prompt.
4. Presence/heartbeat messages over the same LoRa link give a rough
   proximity signal (RSSI-based, not precise-location grade).
