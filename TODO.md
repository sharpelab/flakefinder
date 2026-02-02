
## Camera Streaming Performance

- Position tagging in frame callback causes fps drop: 20.5fps → 12.5fps
- Fix: Move stage.position_um calls out of _on_frame callback
- Options: cache positions, sample less frequently, or use async position events


## Stage Specifications (verified in LAS X)

- **Max velocity: 40 mm/s** (measured ~43 mm/s, at max setting)
- **Acceleration: ~62 mm/s²** (derived from scan data)
- **Accel/decel distance: ~12-13mm each**
- **Accel/decel time: ~690ms**

Stage is already at max speed. Acceleration zones need to be discarded in post-processing.
