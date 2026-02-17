#!/bin/bash
# Fetch speed sweep scans from microscope, stitch at downsample 2, open all

set -e
cd "$(dirname "$0")/.."

SPEEDS=(5 10 15 20 25 30 35 40)

echo "Fetching scans from microscope..."
for speed in "${SPEEDS[@]}"; do
    name="speed_sweep_${speed}mms"
    echo "  $name.zip"
    scp "sharpelab-microscope:flakefinder/scans/${name}.zip" scans/
    unzip -o -d "scans/${name}" "scans/${name}.zip"
done

echo ""
echo "Stitching..."
for speed in "${SPEEDS[@]}"; do
    name="speed_sweep_${speed}mms"
    echo "  $name"
    uv run python stitch_area.py "scans/${name}" --downsample 2 --no-flatfield
done

echo ""
echo "Opening all stitches..."
for speed in "${SPEEDS[@]}"; do
    xdg-open "scans/speed_sweep_${speed}mms_stitch.png" &
    sleep 0.2
done

echo "Done!"
