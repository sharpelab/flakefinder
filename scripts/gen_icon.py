"""Generate FlakeFinder icon: hexagonal flake on dark background."""

import math
from pathlib import Path

from PIL import Image, ImageDraw

SIZE = 256
CENTER = SIZE // 2
BG = (30, 30, 46)  # dark blue-gray
FLAKE_COLOR = (130, 170, 255)  # light blue
ACCENT_COLOR = (180, 130, 255)  # purple accent
OUTLINE_COLOR = (90, 120, 200)


def hex_points(cx, cy, radius, rotation=0):
    """Generate hexagon vertices."""
    return [
        (
            cx + radius * math.cos(math.radians(60 * i + rotation)),
            cy + radius * math.sin(math.radians(60 * i + rotation)),
        )
        for i in range(6)
    ]


def draw_flake(draw, cx, cy, size):
    """Draw a stylized crystalline flake."""
    # Outer hexagon (filled)
    outer = hex_points(cx, cy, size * 0.85, rotation=30)
    draw.polygon(outer, fill=FLAKE_COLOR, outline=OUTLINE_COLOR, width=2)

    # Inner hexagon (darker, gives depth)
    inner = hex_points(cx, cy, size * 0.55, rotation=30)
    inner_color = (100, 140, 220)
    draw.polygon(inner, fill=inner_color, outline=OUTLINE_COLOR, width=1)

    # Center hexagon (accent)
    center = hex_points(cx, cy, size * 0.25, rotation=30)
    draw.polygon(center, fill=ACCENT_COLOR, outline=OUTLINE_COLOR, width=1)

    # Radiating lines from center to outer vertices (crystal structure)
    for i in range(6):
        ox, oy = outer[i]
        draw.line([(cx, cy), (ox, oy)], fill=OUTLINE_COLOR, width=2)


img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Rounded-rect background
draw.rounded_rectangle(
    [(8, 8), (SIZE - 8, SIZE - 8)],
    radius=40,
    fill=BG,
)

draw_flake(draw, CENTER, CENTER, SIZE * 0.42)

# Save as .ico with multiple sizes — pass the 256px image and let PIL downsample
out = Path(__file__).resolve().parent.parent / "src" / "flakefinder" / "assets" / "icon.ico"
ico_sizes = [(s, s) for s in (16, 24, 32, 48, 64, 128, 256)]
img.save(out, format="ICO", sizes=ico_sizes)
print(f"Saved {out} ({out.stat().st_size:,} bytes)")

# Also save a PNG for preview
png_out = "/tmp/flakefinder_icon_preview.png"
img.save(png_out)
print(f"Preview: {png_out}")
