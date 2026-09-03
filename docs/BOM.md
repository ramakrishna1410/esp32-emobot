# Bill of Materials

## Confirmed hardware (per unit — two units planned: parent-bot, kid-bot)

| Part | Role | Notes |
|---|---|---|
| ESP32-S3 dev kit, round display (touch + ambient light sensor, WiFi/BLE, accelerometer/gyro, onboard speaker + mic) | Main board — UI, audio, decision-making | Exact model to confirm in wiring notes once pinout is documented |
| ESP32-CAM module | Vision co-processor | Wired to main board over UART; needs its own 5V supply (camera + WiFi draw more current than the main board's onboard regulator may supply cleanly) |
| LoRa+GPS module (T-Beam class) | Offline bot-to-bot link + outdoor GPS | Confirm exact model for LoRa frequency band / regional compliance |

## Not yet sourced

| Part | Feature it unlocks | Suggested part |
|---|---|---|
| Room-climate sensor | Room "weather" (temp/humidity) query and climate alerts | BME280 (I2C, cheap, widely supported) |
| Fan current-clamp sensor | "Fan left running while away" energy-waste alert | ACS712 or SCT-013 clamp-on current sensor |
| PIR / presence sensor | Local "is anyone home" gate for the energy-waste alert (avoid clamp-sensor-only false positives) | Standard PIR module |

## Open questions

- Exact LoRa+GPS module part number (affects frequency-band compliance and
  whether it's a standalone ESP32 board or a bare module needing its own host).
- Whether both hardware sets (parent-bot + kid-bot, each with all three
  boards) are already on hand or still being assembled.
