# Fracture Detection Model - Integration Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download & Prepare MURA Dataset
```bash
python download_mura.py --data-dir data/mura_raw --output-dir data/fracture_classification
```

### 3. Train Model
```bash
python train_fracture_model.py \
    --data-dir data/fracture_classification \
    --model-name efficientnet_b4 \
    --batch-size 16 \
    --epochs 50 \
    --output-dir models
```

### 4. Export to ONNX
```bash
python export_onnx.py \
    --checkpoint models/checkpoints/best_model.ckpt \
    --model-name efficientnet_b4 \
    --output models/efficientnet_b4_fracture_classifier.onnx \
    --verify --benchmark
```

### 5. Run Complete Pipeline (All Steps)
```bash
python run_pipeline.py --model-name efficientnet_b4 --epochs 50
```

---

## Integration with Existing App

### Option 1: Environment Variable (No Code Changes)
```bash
# Windows PowerShell
$env:USE_CUSTOM_MODEL="true"
python app.py

# Windows CMD
set USE_CUSTOM_MODEL=true
python app.py

# Linux/Mac
export USE_CUSTOM_MODEL=true
python app.py
```

### Option 2: Modify bone_detection_tab.py (Explicit)
```python
# In bone_detection_tab.py __init__ method:
from custom_model_handler import get_model_handler

class BoneDetectionTab(ttk.Frame):
    def __init__(self, parent, db_manager: DatabaseManager, model_handler: ModelHandler = None):
        super().__init__(parent)
        self.db_manager = db_manager
        # Use factory - auto-detects USE_CUSTOM_MODEL env var
        self.model_handler = model_handler or get_model_handler()
        ...
```

### Option 3: Config File
Edit `config.yaml`:
```yaml
USE_CUSTOM_MODEL: true
CUSTOM_MODEL_PATH: models/efficientnet_b4_fracture_classifier.onnx
```

---

## Architecture Options

| Model | Params | Speed | Accuracy | Best For |
|-------|--------|-------|----------|----------|
| `efficientnet_b4` | 19M | Fast | High | Balanced (recommended) |
| `efficientnet_b5` | 30M | Medium | Higher | Maximum accuracy |
| `vit_base_patch16_224` | 86M | Slow | Highest | Research |
| `swin_base_patch4_window7_224` | 88M | Medium | High | Medical imaging |
| `convnext_base` | 88M | Fast | High | Modern CNN |

---

## Dataset Structure (Expected)

```
data/fracture_classification/
├── train/
│   ├── transverse/*.png
│   ├── oblique/*.png
│   ├── compound/*.png
│   └── stress/*.png
└── val/
    ├── transverse/*.png
    ├── oblique/*.png
    ├── compound/*.png
    └── stress/*.png
```

---

## MURA to Fracture Type Mapping

| MURA Body Part | Mapped Fracture Types |
|----------------|----------------------|
| XR_ELBOW | transverse, oblique |
| XR_FINGER | transverse, oblique, stress |
| XR_FOREARM | transverse, oblique, compound |
| XR_HAND | transverse, oblique, stress |
| XR_HUMERUS | transverse, oblique, compound |
| XR_SHOULDER | transverse, oblique |
| XR_WRIST | transverse, oblique, stress, compound |

---

## Verification Checklist

After training, verify:
- [ ] ONNX model loads without errors
- [ ] Numerical parity: PyTorch vs ONNX < 1e-4 difference
- [ ] Inference speed < 100ms per image (CPU)
- [ ] All 4 fracture types predict correctly on test_images/
- [ ] `custom_model_handler.py` passes `test_integration.py`
- [ ] App runs with `USE_CUSTOM_MODEL=true`

---

## Troubleshooting

**Model not found error:**
```
FileNotFoundError: ONNX model not found at models/efficientnet_b4_fracture_classifier.onnx
```
→ Run export_onnx.py first, or check model path in config.yaml

**CUDA out of memory:**
```bash
# Reduce batch size
python train_fracture_model.py --batch-size 8
```

**Low accuracy:**
- Increase epochs: `--epochs 100`
- Try larger model: `--model-name efficientnet_b5`
- Check class balance in dataset

**Import errors:**
```bash
pip install -r requirements.txt --upgrade
```

---

## Files Created

| File | Purpose |
|------|---------|
| `requirements.txt` | Python dependencies |
| `config.yaml` | Configuration file |
| `download_mura.py` | Download & organize MURA dataset |
| `train_fracture_model.py` | PyTorch Lightning training script |
| `export_onnx.py` | Export to ONNX + verify |
| `custom_model_handler.py` | Drop-in replacement for ModelHandler |
| `test_integration.py` | Verify interface compatibility |
| `run_pipeline.py` | Run complete pipeline |
| `INTEGRATION_GUIDE.md` | This file |

---

## Model Performance Targets

| Metric | Target |
|--------|--------|
| Validation Accuracy | > 85% |
| Per-class Accuracy | > 80% |
| Inference Time (CPU) | < 100ms |
| ONNX Size | < 100MB |
| Numerical Parity | < 1e-4 max diff |