// Shared response-engine interface, used by main-board firmware.
//
// Scaffold only — see docs/architecture.md "Message flow: mood detection to
// response" for the intended behavior: cloud-LLM-first when WiFi is up,
// falling back to a local scripted table (per TriggerCategory) when offline.
//
// Deliberately hardware-agnostic so the selection logic (no-repeat, category
// lookup) can be unit-tested on desktop — see /tools.

#pragma once

enum class TriggerCategory {
  MoodSad,
  MoodHappy,
  LifeEvent,
  ClimateAlert,
  EnergyWasteAlert,
  KidCalledHome,
  IdleAmbient,
};

struct ResponseResult {
  const char *text;
  bool from_llm; // false when served from the local scripted fallback
};

// TODO(Phase 1): implement WiFi check -> cloud LLM request -> scripted
// fallback. Keep the scripted table small and stored in flash
// (SPIFFS/LittleFS) per docs/architecture.md.
ResponseResult GetResponse(TriggerCategory category, const char *context);
