
## Camera Streaming Performance

- Position tagging in frame callback causes fps drop: 20.5fps → 12.5fps
- Fix: Move stage.position_um calls out of _on_frame callback
- Options: cache positions, sample less frequently, or use async position events

