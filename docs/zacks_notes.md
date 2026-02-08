can't move to coordinates defined by bounding boxes, they don't quite work

● Chip 1 data from chips_5x_stitch_chips.json:
  - bbox: x: 14895-48451, y: 15731-30929 µm
  - centroid: (31676, 23316) µm

  At 20x with 3x3 binning: frame is ~1824×1216 px @ 0.36 µm/px = 657×438 µm

  Calculated margins:
  - X: 2000 µm each side → x: 16895 to 46451
  - Y: half frame (~220 µm) each side → y: 15951 to 30709

  python scan_area_v1.py -o chip1_20x \
    --objective-mag 20x \
    --area-rect 16895,46451,15951,30709 \
    --speed-mm 20 \
    --auto-focus-pos 31676,23316





● Frame at 20x 3x3: 657×438 µm, Y step with 12% overlap: ~385 µm

  For 4 rows centered on centroid:
  - Y range: ~1.5 steps each side ≈ ±800 µm from centroid Y (23316)
  - Y: 22500 to 24100

  Half X width centered on centroid (31676):
  - Original width: ~29500 µm → half: ~14750 µm
  - X: 24300 to 39050

  python scan_area_v1.py -o chip1_20x_test \
    --objective-mag 20x \
    --area-rect 24300,39050,22500,24100 \
    --speed-mm 20 \
    --auto-focus-pos 31676,23316