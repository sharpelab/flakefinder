# Background

Context on 2D material exfoliation and flake detection requirements.

## Exfoliation Process

Mechanical exfoliation to produce 2D material flakes on substrates.

### Substrate Selection
- SiO2/Si wafers
- Oxide thickness matters
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
- Optical microscopy to locate flakes
- Reject tape residue and debris

---

## Target Materials

What the flake finder should detect:

| Material | Thickness | Notes |
|----------|-----------|-------|
| Graphene | 1-8 layers | |
| Graphite | ~5 nm | |
| WSe2 | 1-8 layers | |
| hBN thin | 1-4 layers | Monolayer hBN is hardest |
| hBN thick | 5-40 nm | Should estimate thickness with ~10% error |

Easy to add new materials.

## Current Detection Status

What the existing 2DMatGMM system handles:

| Material | Status |
|----------|--------|
| Graphene | 1-4 layers working |
| WSe2 | Unclear |
| hBN | Algorithm is bad |
