# EmoBot — Offline-First ESP32 Companion Toy

An emotion-aware companion toy built on ESP32-S3, designed to work fully offline
for its core loop, with an optional cloud-LLM layer for natural conversation
when WiFi is available.

Two physical units are planned: `parent-bot` and `kid-bot`, sharing the same
hardware and firmware, differing only by build-time role config.

## Hardware (per unit)

- **ESP32-S3 dev kit with round display** — touch input, ambient light sensor,
  WiFi/BLE, accelerometer/gyro, onboard speaker + mic. This is the main board:
  owns the UI, audio, and decision-making.
- **ESP32-CAM module** (wired in separately) — vision co-processor for
  camera-based emotion sensing, UART-linked to the main board.
- **LoRa+GPS module** — offline long-range bot-to-bot link (presence,
  "come home" signaling) plus outdoor GPS for distance/direction once outside.

See [docs/BOM.md](docs/BOM.md) for the full parts list and open sourcing items,
and [docs/architecture.md](docs/architecture.md) for how the three chips
cooperate and how online/offline modes are handled.

## Repo layout

```
/firmware/common/        shared: response-engine, face-renderer, ble-link, lora-link
/firmware/main-board/    round-display board firmware (parent-bot / kid-bot via build config)
/firmware/cam-coproc/    ESP32-CAM firmware (face detection + emotion classifier)
/tools/                  desktop scripts (e.g. simulate response-engine selection logic)
/docs/                   BOM, architecture notes, wiring, roadmap
```

Build system: [PlatformIO](https://platformio.org/), Arduino framework for
ESP32-S3 targets, per-role environments in each `platformio.ini`.

## Status

Early scaffold stage — see [docs/roadmap.md](docs/roadmap.md) for the phased
build plan (Phase 1 core companion, Phase 1B camera emotion, Phase 2 twin-bot
pairing, Phase 3 stretch goals). No firmware logic is implemented yet; this
repo currently holds the project structure and PlatformIO skeletons to build
against.
