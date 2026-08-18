#!/usr/bin/env python3
"""
Complete Training Pipeline Runner
Orchestrates: Download MURA -> Prepare Data -> Train Model -> Export ONNX -> Verify
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_cmd(cmd: list, cwd: str = None, env: dict = None) -> bool:
    """Run command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    
    result = subprocess.run(cmd, cwd=cwd, env=merged_env)
    return result.returncode == 0


def check_dependencies():
    """Check if required packages are installed."""
    required = [
        'torch', 'torchvision', 'pytorch_lightning', 'timm',
        'albumentations', 'onnx', 'onnxruntime', 'cv2', 'sklearn'
    ]
    
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"Missing packages: {missing}")
        print("Install with: pip install -r requirements.txt")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Complete fracture detection training pipeline")
    parser.add_argument("--skip-download", action="store_true", help="Skip MURA download")
    parser.add_argument("--skip-train", action="store_true", help="Skip training")
    parser.add_argument("--skip-export", action="store_true", help="Skip ONNX export")
    parser.add_argument("--model-name", default="efficientnet_b4", 
                       choices=["efficientnet_b4", "efficientnet_b5", "vit_base_patch16_224", 
                               "swin_base_patch4_window7_224", "convnext_base"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data-dir", default="data/fracture_classification")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    parser.add_argument("--fast-dev-run", action="store_true", help="Quick test run")
    parser.add_argument("--use-gdrive", action="store_true", help="Use Google Drive for MURA download")
    args = parser.parse_args()
    
    project_root = Path(__file__).parent
    
    if not check_dependencies():
        sys.exit(1)
    
    print("="*60)
    print("FRACTURE DETECTION MODEL TRAINING PIPELINE")
    print("="*60)
    print(f"Model: {args.model_name}")
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print("="*60)
    
    # Step 1: Download and prepare MURA dataset
    if not args.skip_download:
        print("\n[STEP 1/4] Downloading and preparing MURA dataset...")
        cmd = [
            sys.executable, "download_mura.py",
            "--data-dir", "data/mura_raw",
            "--output-dir", args.data_dir,
        ]
        if args.use_gdrive:
            cmd.append("--use-gdrive")
        
        if not run_cmd(cmd, cwd=project_root):
            print("ERROR: Dataset preparation failed")
            sys.exit(1)
    else:
        print("\n[STEP 1/4] Skipping dataset download (--skip-download)")
    
    # Verify dataset exists
    data_path = project_root / args.data_dir
    if not data_path.exists():
        print(f"ERROR: Dataset not found at {data_path}")
        sys.exit(1)
    
    # Check class directories
    for split in ['train', 'val']:
        for cls in ["transverse", "oblique", "compound", "stress"]:
            cls_dir = data_path / split / cls
            if not cls_dir.exists() or len(list(cls_dir.glob("*.png"))) == 0:
                print(f"WARNING: {cls_dir} is empty or missing")
    
    # Step 2: Train model
    if not args.skip_train:
        print("\n[STEP 2/4] Training model...")
        cmd = [
            sys.executable, "train_fracture_model.py",
            "--data-dir", args.data_dir,
            "--model-name", args.model_name,
            "--batch-size", str(args.batch_size),
            "--img-size", str(args.img_size),
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--output-dir", args.output_dir,
        ]
        if args.resume:
            cmd.extend(["--resume", args.resume])
        if args.fast_dev_run:
            cmd.append("--fast-dev-run")
        
        if not run_cmd(cmd, cwd=project_root):
            print("ERROR: Training failed")
            sys.exit(1)
    else:
        print("\n[STEP 2/4] Skipping training (--skip-train)")
    
    # Step 3: Export to ONNX
    if not args.skip_export:
        print("\n[STEP 3/4] Exporting to ONNX...")
        
        # Find best checkpoint
        checkpoint_dir = Path(args.output_dir) / "checkpoints"
        checkpoints = list(checkpoint_dir.glob("*.ckpt"))
        best_ckpt = None
        for ckpt in checkpoints:
            if "best" in ckpt.name or "last" in ckpt.name:
                best_ckpt = ckpt
                break
        
        if best_ckpt is None and checkpoints:
            best_ckpt = sorted(checkpoints, key=lambda x: x.stat().st_mtime)[-1]
        
        if best_ckpt is None:
            print("ERROR: No checkpoint found")
            sys.exit(1)
        
        print(f"Using checkpoint: {best_ckpt}")
        
        cmd = [
            sys.executable, "export_onnx.py",
            "--checkpoint", str(best_ckpt),
            "--model-name", args.model_name,
            "--output", str(Path(args.output_dir) / f"{args.model_name}_fracture_classifier.onnx"),
            "--img-size", str(args.img_size),
            "--verify",
            "--benchmark",
            "--simplify",
        ]
        
        if not run_cmd(cmd, cwd=project_root):
            print("ERROR: ONNX export failed")
            sys.exit(1)
    else:
        print("\n[STEP 3/4] Skipping ONNX export (--skip-export)")
    
    # Step 4: Test integration
    print("\n[STEP 4/4] Testing integration with custom_model_handler...")
    cmd = [sys.executable, "custom_model_handler.py"]
    if not run_cmd(cmd, cwd=project_root):
        print("WARNING: Integration test failed, but model may still work")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"Model saved to: {args.output_dir}/")
    print(f"ONNX model: {args.output_dir}/{args.model_name}_fracture_classifier.onnx")
    print(f"Class info: {args.output_dir}/class_info.json")
    print("\nTo use in the app:")
    print("  1. Set environment variable: USE_CUSTOM_MODEL=true")
    print("  2. Or modify bone_detection_tab.py to use CustomModelHandler")
    print("="*60)


if __name__ == "__main__":
    main()