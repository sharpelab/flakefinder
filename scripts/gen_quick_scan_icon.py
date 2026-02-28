"""Generate Quick Scan icon: tiled mosaic of offset frames on dark background."""

from pathlib import Path

from PIL import Image, ImageDraw

SIZE = 256
CENTER = SIZE // 2
BG = (30, 30, 46)  # dark blue-gray (matches flakefinder icon)

# Tile colors — varying blues, last tile is accent (just placed)
TILE_COLORS = [
    (70, 110, 180),  # steel blue
    (80, 125, 195),
    (65, 100, 170),
    (85, 130, 200),
    (75, 115, 185),
    (60, 95, 165),
    (90, 135, 205),
    (70, 108, 178),
    (140, 170, 255),  # bright accent — the "latest" frame
]
TILE_OUTLINE = (50, 80, 140)
GRID_ROWS = 3
GRID_COLS = 3

img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Rounded-rect background
draw.rounded_rectangle([(8, 8), (SIZE - 8, SIZE - 8)], radius=40, fill=BG)

# Tile grid area (inset from background edges)
margin = 42
area_w = SIZE - 2 * margin
area_h = SIZE - 2 * margin
tile_w = area_w / GRID_COLS
tile_h = area_h / GRID_ROWS
gap = 4  # gap between tiles

# Draw tiles with slight stagger — later tiles overlap earlier ones
for idx in range(GRID_ROWS * GRID_COLS):
    row = idx // GRID_COLS
    col = idx % GRID_COLS
    color = TILE_COLORS[idx]

    # Slight offset per tile to give a "placed" feel
    ox = (idx % 3 - 1) * 2
    oy = (idx // 3 - 1) * 1.5

    x0 = margin + col * tile_w + gap / 2 + ox
    y0 = margin + row * tile_h + gap / 2 + oy
    x1 = x0 + tile_w - gap
    y1 = y0 + tile_h - gap

    draw.rounded_rectangle(
        [(x0, y0), (x1, y1)],
        radius=6,
        fill=color,
        outline=TILE_OUTLINE,
        width=2,
    )

# Save .ico with multiple sizes
out = Path(__file__).resolve().parent.parent / "src" / "quick_scan" / "assets" / "icon.ico"
ico_sizes = [(s, s) for s in (16, 24, 32, 48, 64, 128, 256)]
img.save(out, format="ICO", sizes=ico_sizes)
print(f"Saved {out} ({out.stat().st_size:,} bytes)")

# Also save a PNG for preview
png_out = "/tmp/quick_scan_icon_preview.png"
img.save(png_out)
print(f"Preview: {png_out}")
