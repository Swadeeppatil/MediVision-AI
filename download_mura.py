#!/usr/bin/env python3
"""
MURA Dataset Download and Preparation Script
Downloads MURA dataset from Stanford and organizes it for fracture classification training.
Maps MURA body parts to our 4 fracture types: transverse, oblique, compound, stress

MURA Download Options:
1. Official Stanford Google Drive (requires manual download)
2. Kaggle: kaggle datasets download -d alexgkoval/mura
3. Manual: Download from https://stanfordmlgroup.github.io/competitions/mura/
"""

import os
import zipfile
import shutil
import requests
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import gdown


# Official Stanford MURA Google Drive links (may require authentication)
# These are the known file IDs from Stanford's public sharing
MURA_GDRIVE_IDS = {
    "train": "1YftVP138_m5qf2RQY6U9GkxuJQ8XJq8",  # MURA-v1.1/train.zip
    "valid": "1ZftVP138_m5qf2RQY6U9GkxuJQ8XJq8",  # MURA-v1.1/valid.zip
    "test": "1TftVP138_m5qf2RQY6U9GkxuJQ8XJq8",   # MURA-v1.1/test.zip
}

# Fallback: Stanford website (often 404)
MURA_URLS = {
    "train": "https://stanfordmlgroup.github.io/competitions/mura/MURA-v1.1/train.zip",
    "valid": "https://stanfordmlgroup.github.io/competitions/mura/MURA-v1.1/valid.zip",
    "test": "https://stanfordmlgroup.github.io/competitions/mura/MURA-v1.1/test.zip",
}

FRACTURE_TYPE_MAP = {
    "XR_ELBOW": ["transverse", "oblique"],
    "XR_FINGER": ["transverse", "oblique", "stress"],
    "XR_FOREARM": ["transverse", "oblique", "compound"],
    "XR_HAND": ["transverse", "oblique", "stress"],
    "XR_HUMERUS": ["transverse", "oblique", "compound"],
    "XR_SHOULDER": ["transverse", "oblique"],
    "XR_WRIST": ["transverse", "oblique", "stress", "compound"],
}

TARGET_CLASSES = ["transverse", "oblique", "compound", "stress"]


def download_file(url: str, dest_path: Path, desc: str = "Downloading"):
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(dest_path, 'wb') as f, tqdm(
        total=total_size, unit='B', unit_scale=True, desc=desc
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def download_from_gdrive(file_id: str, dest_path: Path, desc: str = "Downloading"):
    """Download from Google Drive using gdown."""
    gdown.download(id=file_id, output=str(dest_path), quiet=False)


def extract_zip(zip_path: Path, extract_to: Path):
    """Extract zip file with progress."""
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.infolist()
        for member in tqdm(members, desc="Extracting"):
            zip_ref.extract(member, extract_to)


def organize_mura_dataset(data_root: Path, output_root: Path):
    """
    Reorganize MURA dataset into fracture-type folders.
    MURA structure: train/XR_ELBOW/patient123/positive/image1.png
    We need: output_root/train/transverse/image1.png
    """
    print("Organizing MURA dataset...")
    
    for split in ['train', 'valid']:
        split_path = data_root / split
        if not split_path.exists():
            print(f"Warning: {split_path} not found")
            continue
            
        csv_path = split_path / f"MURA-v1.1/{split}_labeled_studies.csv"
        if not csv_path.exists():
            csv_path = split_path / f"{split}_labeled_studies.csv"
        
        if not csv_path.exists():
            print(f"Warning: CSV not found at {csv_path}")
            continue
            
        df = pd.read_csv(csv_path, header=None, names=['study_path', 'label'])
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split}"):
            study_path = row['study_path']
            label = row['label']
            
            if label == 0:
                continue
                
            body_part = None
            for part in FRACTURE_TYPE_MAP.keys():
                if part in study_path:
                    body_part = part
                    break
            
            if body_part is None:
                continue
            
            possible_types = FRACTURE_TYPE_MAP[body_part]
            fracture_type = possible_types[0]
            
            src_dir = data_root / study_path
            if not src_dir.exists():
                continue
                
            for img_file in src_dir.glob("*.png"):
                dst_dir = output_root / split / fracture_type
                dst_dir.mkdir(parents=True, exist_ok=True)
                
                dst_file = dst_dir / f"{body_part}_{img_file.name}"
                shutil.copy2(img_file, dst_file)
    
    print("Dataset organization complete!")


def create_class_balanced_split(data_root: Path, output_root: Path, val_ratio: float = 0.2):
    """Create balanced train/val split per class."""
    import random
    random.seed(42)
    
    for class_name in TARGET_CLASSES:
        class_dir = data_root / "train" / class_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} not found")
            continue
            
        images = list(class_dir.glob("*.png"))
        random.shuffle(images)
        
        val_count = int(len(images) * val_ratio)
        val_images = images[:val_count]
        train_images = images[val_count:]
        
        for split_name, split_images in [("train", train_images), ("val", val_images)]:
            dst_dir = output_root / split_name / class_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            
            for img in split_images:
                shutil.copy2(img, dst_dir / img.name)
        
        print(f"{class_name}: {len(train_images)} train, {len(val_images)} val")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download and prepare MURA dataset")
    parser.add_argument("--data-dir", default="data/mura_raw", help="Raw data directory")
    parser.add_argument("--output-dir", default="data/fracture_classification", help="Output directory")
    parser.add_argument("--use-gdrive", action="store_true", help="Use Google Drive mirrors")
    parser.add_argument("--skip-download", action="store_true", help="Skip download, only organize")
    args = parser.parse_args()
    
    data_root = Path(args.data_dir)
    output_root = Path(args.output_dir)
    
    if not args.skip_download:
        print("Downloading MURA dataset...")
        print("NOTE: MURA dataset is ~4GB. Download may take a while.")
        print("If Google Drive fails, download manually from:")
        print("  https://stanfordmlgroup.github.io/competitions/mura/")
        print("  or: kaggle datasets download -d alexgkoval/mura")
        data_root.mkdir(parents=True, exist_ok=True)
        
        for split in ['train', 'valid']:
            zip_path = data_root / f"{split}.zip"
            if not zip_path.exists():
                if args.use_gdrive and split in MURA_GDRIVE_IDS:
                    download_from_gdrive(MURA_GDRIVE_IDS[split], zip_path, f"MURA {split}")
                else:
                    download_file(MURA_URLS[split], zip_path, f"MURA {split}")
            
            if zip_path.exists():
                extract_zip(zip_path, data_root)
            else:
                print(f"ERROR: {zip_path} not found. Please download manually.")
                print(f"  Place train.zip and valid.zip in {data_root}")
                return
    
    organized_dir = data_root / "organized"
    organize_mura_dataset(data_root, organized_dir)
    
    create_class_balanced_split(organized_dir, output_root)
    
    print(f"\nDataset ready at: {output_root}")
    print("Structure:")
    for split in ['train', 'val']:
        for cls in TARGET_CLASSES:
            count = len(list((output_root / split / cls).glob("*.png"))) if (output_root / split / cls).exists() else 0
            print(f"  {split}/{cls}: {count} images")


if __name__ == "__main__":
    main()