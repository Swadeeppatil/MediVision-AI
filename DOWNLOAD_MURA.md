# MURA Dataset Download Instructions

The MURA dataset (40,000+ musculoskeletal X-rays) is required for training. Stanford's direct links are often unavailable.

## Option 1: Kaggle (Recommended - Easiest)

```bash
# 1. Install Kaggle CLI
pip install kaggle

# 2. Set up credentials (get from kaggle.com/settings)
# Place kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME/KAGGLE_KEY env vars

# 3. Download
kaggle datasets download -d alexgkoval/mura -p data/mura_raw

# 4. Extract
cd data/mura_raw
unzip mura.zip
# This creates MURA-v1.1/train/ and MURA-v1.1/valid/
```

## Option 2: Manual Download from Stanford

1. Go to: https://stanfordmlgroup.github.io/competitions/mura/
2. Click "Download Dataset" 
3. Download `train.zip` and `valid.zip`
4. Place in `data/mura_raw/`
5. Extract:
   ```bash
   cd data/mura_raw
   unzip train.zip
   unzip valid.zip
   ```

## Option 3: Google Drive (Unreliable)

The official Stanford Google Drive links sometimes work:
- train.zip: https://drive.google.com/uc?id=1YftVP138_m5qf2RQY6U9GkxuJQ8XJq8
- valid.zip: https://drive.google.com/uc?id=1ZftVP138_m5qf2RQY6U9GkxuJQ8XJq8

## After Download

Run the preparation script:
```bash
python download_mura.py --data-dir data/mura_raw --output-dir data/fracture_classification --skip-download
```

This will:
1. Read MURA CSV labels
2. Map body parts to fracture types (transverse, oblique, compound, stress)
3. Create balanced train/val splits
4. Output to `data/fracture_classification/`

## Expected Structure After Preparation

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

## Quick Test Without Full Dataset

For quick testing, you can create synthetic data:
```bash
python create_synthetic_data.py
```

Then train with `--fast-dev-run`:
```bash
python train_fracture_model.py --fast-dev-run --data-dir data/fracture_classification --epochs 1
```