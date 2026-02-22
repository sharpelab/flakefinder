# Background

Context on 2D material exfoliation and why automated flake detection matters.

## Exfoliation Process

Mechanical exfoliation to produce 2D material flakes on substrates.

### Substrate Selection
- SiO2/Si wafers
- Oxide thickness matters — 90nm SiO2 is standard for graphene and hBN
  (optimal thin-film interference contrast in R and G channels)
- Oxide growth method: dry vs wet

### Substrate Prep
- Plasma treatment to improve adhesion

### Source Material
- Material choice (graphene, hBN, WSe2, etc.)
- Crystal type and quality

### Tape Prep
- Choice of tape
- How you "populate" the tape with material
- How you press tape onto substrate
- Temperature during pressing

### Post-Tape Treatment
- Heat anneal

### Peeling
- Speed of peel
- Angle of peeling
- Hot peel vs cold peel

### Finding Flakes
- Optical microscopy to locate flakes on the substrate
- Reject tape residue and debris
- This is what FlakeFinder automates

## Downstream Tasks

What happens after flakes are found — this is why detection speed and quality matter.

### Extraction & Stacking
- Promising flakes are picked up from the substrate using a polymer stamp
- Flakes are stacked into heterostructures (e.g., hBN/graphene/hBN sandwiches)
- Alignment matters — crystallographic orientation, relative twist angle
- A single device might require 3-5 individually exfoliated flakes

### Device Fabrication
- Patterning via e-beam lithography
- Metal contact deposition
- Etching to define device geometry

### Measurement
- Cryogenic transport measurements (dilution fridge, ~10 mK)
- Quantum Hall effect, superconductivity, ferromagnetism
- Each device represents weeks of cumulative effort

### Why Automation Matters
- Exfoliation yield is low — most of a chip is empty substrate
- Manual microscope search takes 30-60 minutes per chip
- FlakeFinder scans a full wafer in ~15 minutes, detecting flakes the
  operator would spend hours finding by eye
- Higher throughput = more devices = faster research iteration

## Target Materials

| Material | Thickness Range | Substrate | Detection | Notes |
|----------|----------------|-----------|-----------|-------|
| hBN thin | 1-4 layers (< 15 nm) | 90nm SiO2 | Working (10x/20x) | Cal curve from AFM-verified samples |
| hBN thick | 5-50 nm | 90nm SiO2 | Working (10x/20x) | Thickness estimated from contrast projection |
| Graphene | 1-8 layers | 90nm SiO2 | Partial | Cal curve structure known, needs lab calibration data |
| WSe2 | 1-8 layers | 90nm SiO2 | Not yet | Similar contrast approach expected |

See `segmentation.py` `DetectorConfig.from_material()` for current presets.

## Key References

- `docs/bn_thickness_calibration.md` — hBN contrast-to-thickness data
- `docs/graphene_contrast_model.md` — graphene contrast analysis from 2DMatGMM
