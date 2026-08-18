#!/usr/bin/env python3
"""
Create Synthetic Fracture Data for Quick Testing
Generates dummy X-ray images for each fracture class to test the training pipeline.
"""

import os
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm


CLASSES = ["transverse", "oblique", "compound", "stress"]
NUM_TRAIN_PER_CLASS = 50
NUM_VAL_PER_CLASS = 10
IMG_SIZE = 224


def create_synthetic_xray(class_name: str, idx: int) -> np.ndarray:
    """Create a synthetic X-ray-like image with class-specific patterns."""
    # Base: dark background with bone-like structures
    img = np.random.randint(20, 50, (IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    
    # Add "bone" structure - bright curved lines
    center = IMG_SIZE // 2
    
    if class_name == "transverse":
        # Horizontal line across middle (transverse fracture)
        cv2.line(img, (50, center), (IMG_SIZE-50, center), 255, 3)
        cv2.line(img, (50, center+2), (IMG_SIZE-50, center+2), 0, 1)  # fracture gap
        
    elif class_name == "oblique":
        # Diagonal line
        cv2.line(img, (50, IMG_SIZE-50), (IMG_SIZE-50, 50), 255, 3)
        cv2.line(img, (52, IMG_SIZE-48), (IMG_SIZE-48, 52), 0, 1)  # fracture gap
        
    elif class_name == "compound":
        # Multiple fragments
        cv2.line(img, (50, center), (center-20, center), 255, 3)
        cv2.line(img, (center+20, center), (IMG_SIZE-50, center), 255, 3)
        # Fragment displacement
        cv2.circle(img, (center, center), 15, 255, -1)
        
    elif class_name == "stress":
        # Thin hairline crack
        cv2.line(img, (center-40, center), (center+40, center), 180, 1)
        # Micro-cracks
        for i in range(5):
            x = center - 30 + i * 15
            cv2.line(img, (x, center-10), (x+5, center+10), 150, 1)
    
    # Add noise and texture
    noise = np.random.randint(0, 30, (IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    img = cv2.add(img, noise)
    
    # Add some bone-like texture (random bright spots)
    for _ in range(20):
        x, y = np.random.randint(20, IMG_SIZE-20, 2)
        cv2.circle(img, (x, y), np.random.randint(2, 8), 
                   np.random.randint(100, 255), -1)
    
    # Convert to 3-channel
    img_3ch = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img_3ch


def main():
    output_dir = Path("data/fracture_classification")
    
    print("Creating synthetic fracture dataset...")
    
    for split, num_per_class in [("train", NUM_TRAIN_PER_CLASS), ("val", NUM_VAL_PER_CLASS)]:
        for class_name in CLASSES:
            class_dir = output_dir / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            for i in tqdm(range(num_per_class), desc=f"{split}/{class_name}"):
                img = create_synthetic_xray(class_name, i)
                filename = f"{class_name}_{split}_{i:04d}.png"
                cv2.imwrite(str(class_dir / filename), img)
    
    print(f"\nSynthetic dataset created at: {output_dir}")
    print("Structure:")
    for split in ["train", "val"]:
        for class_name in CLASSES:
            count = len(list((output_dir / split / class_name).glob("*.png")))
            print(f"  {split}/{class_name}: {count} images")
    
    print("\nNow you can test training:")
    print("  python train_fracture_model.py --fast-dev-run --data-dir data/fracture_classification --epochs 1")


if __name__ == "__main__":
    main()