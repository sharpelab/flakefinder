# MaskTerial API Reference

Local server for 2D material flake detection using Mask2Former segmentation + Adaptive Mahalanobis Metric (AMM) classification.

**Repo:** `~/code/MaskTerial`
**Start:** `cd ~/code/MaskTerial && uv run uvicorn server:app --host 0.0.0.0 --port 8000`

## Endpoints

### `GET /`
Health check. Returns `"No Models are loaded..."` when idle.

### `GET /status`
Returns `"Ready for inference"`.

### `GET /available_models`
Lists installed models by type:
```json
{
  "available_models": {
    "classification_models": { "AMM": ["GrapheneH", "hBN_Thin"] },
    "segmentation_models": { "M2F": ["GrapheneH", "hBN_Thin", "Synthetic_Data"] },
    "postprocessing_models": {}
  }
}
```

### `POST /predict`
Main inference endpoint. Multipart form data.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `files` | file(s) | required | Image file(s) to analyze |
| `segmentation_model` | string | null | M2F model name, e.g. `M2F-hBN_Thin` |
| `classification_model` | string | null | AMM model name, e.g. `AMM-hBN_Thin` |
| `postprocessing_model` | string | null | GMM model (not used) |
| `score_threshold` | float | 0.0 | Min segmentation confidence (0-1) |
| `size_threshold` | int | 300 | Min detection size in pixels |
| `min_class_occupancy` | float | 0.0 | Min fraction of pixels assigned to a material class |
| `return_bbox` | bool | false | Include bounding box in response |

**Example:**
```python
import requests
with open('image.png', 'rb') as f:
    r = requests.post(
        'http://localhost:8000/predict',
        files={'files': ('image.png', f, 'image/png')},
        data={
            'segmentation_model': 'M2F-hBN_Thin',
            'classification_model': 'AMM-hBN_Thin',
            'score_threshold': '0.0',
            'size_threshold': '300',
            'return_bbox': 'true',
        },
        timeout=60,
    )
result = r.json()
# Result is list-of-lists when multiple files; [0] for single file
if isinstance(result, list) and result and isinstance(result[0], list):
    result = result[0]
```

**Response:** List of detection dicts:
```json
{
  "mask": {"size": [1216, 1824], "counts": "...RLE..."},
  "thickness": 1,
  "size": 1001,
  "mean_contrast": [0, 0, 0],
  "center": [886, 1108],
  "max_sidelength": 90,
  "min_sidelength": 14,
  "aspect_ratio": 6.1,
  "false_positive_probability": 0.315,
  "entropy": -1.0,
  "bbox": [841, 1088, 83, 44]
}
```

| Field | Description |
|-------|-------------|
| `mask` | RLE-encoded binary mask (pycocotools format) |
| `thickness` | AMM material class (0=background, 1+=material classes) |
| `size` | Detection area in pixels |
| `mean_contrast` | Per-channel contrast `(pixel/bg_mode - 1)` averaged over detection |
| `center` | `[x, y]` centroid |
| `max_sidelength` / `min_sidelength` | Bounding box dimensions in µm (from model config) |
| `aspect_ratio` | max/min sidelength |
| `false_positive_probability` | AMM Mahalanobis distance mapped to probability (lower = more likely real) |
| `entropy` | Classification entropy (-1 if no classification) |
| `bbox` | `[x, y, width, height]` (only with `return_bbox=true`) |

### `POST /upload/m2f`
Upload a custom M2F segmentation model. Fields: `model_name`, `model_file` (.pth), `config_file` (.yaml).

### `POST /upload/amm`
Upload a custom AMM classification model. Fields: `model_name`, `metadata_file`, `loc_file`, `cov_file`, `weights_file`.

### `POST /train/m2f`
Fine-tune M2F model. Fields: `model_name`, `dataset_file` (.zip), `config_file` (.yaml).

### `POST /delete_model`
Delete a loaded model.

### `GET /download_model`
Download a model's files.

## Models

### Segmentation (M2F)
Binary foreground/background segmentation using Mask2Former. Per-material fine-tuning adapts to lab-specific imaging, not material physics.

- **M2F-hBN_Thin**: Trained on thin (1-3L) hBN flakes. Deliberately ignores thick flakes.
- **M2F-GrapheneH**: Trained on thick graphene.
- **M2F-Synthetic_Data**: Base model, nearly useless without fine-tuning.

All models share normalization: `PIXEL_MEAN` BGR [107, 148, 85], `PIXEL_STD` [33, 34, 40]. Detectron2 resizes longest side to 1333px.

Training images: 1920×1200, bluish-gray substrate (mode RGB ~[148, 154, 174]).

### Classification (AMM)
Per-pixel classification using Adaptive Mahalanobis Metric in contrast space.

**Contrast normalization:** `contrast = image / histogram_mode - 1` per channel. Background estimated from histogram peak in [20, 230] range ±5 radius.

**Critical requirement:** Substrate brightness must be ~150-170/255 for AMM to work. At low brightness (~30/255), quantization noise in 8-bit images produces 3-7% contrast noise per level — AMM classifies noise as material.

- **AMM-hBN_Thin**: Classifies thin hBN by thickness class.
- **AMM-GrapheneH**: Classifies thick graphene.

## Additional Models (Zenodo)

Downloadable from MaskTerial's Zenodo releases:
- Synthetic, GrapheneL, GrapheneM, WSe2, WSe2L, MoSe2, WS2

## Known Issues

1. **Low brightness kills AMM** — substrate at ~30/255 (our 0.25ms/gain 4 scans) makes `mean_contrast` return `[0,0,0]` and FP probabilities cluster at 0.98+.
2. **M2F-hBN_Thin ignores thick flakes by design** — training data is thin (1-3L) only.
3. **No NMS** — overlapping M2F detections returned as-is. Multiple masks can cover the same region.
4. **Response wrapping** — single-file predictions return `[[detections]]` (list of lists). Unwrap with `result[0]`.
