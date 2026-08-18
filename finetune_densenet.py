#!/usr/bin/env python3
"""
Fine-tune Existing DenseNet169 Model on Fracture Data
Uses the SAME architecture as model_handler.py for seamless integration.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json


# Same as model_handler.py
FRACTURE_TYPES = ["transverse", "oblique", "compound", "stress"]
NUM_CLASSES = len(FRACTURE_TYPES)
CLASS_TO_IDX = {c: i for i, c in enumerate(FRACTURE_TYPES)}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FractureDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, split: str, transform=None, img_size: int = 224):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.img_size = img_size
        self.samples = []
        
        for class_name in FRACTURE_TYPES:
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*.png"):
                    self.samples.append((str(img_path), CLASS_TO_IDX[class_name]))
        
        print(f"{split}: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label


def get_transforms(img_size: int = 224, is_train: bool = True):
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.OneOf([
                A.GaussNoise(std=0.1),
                A.GaussianBlur(blur_limit=3),
                A.MotionBlur(blur_limit=3),
            ], p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.CLAHE(clip_limit=2.0),
                A.RandomGamma(gamma_limit=(80, 120)),
            ], p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])


class DenseNetFractureModel(pl.LightningModule):
    """Same architecture as model_handler.py but trainable."""
    
    def __init__(self, lr: float = 1e-4, weight_decay: float = 1e-5, unfreeze_layers: int = 3):
        super().__init__()
        self.save_hyperparameters()
        
        # Create DenseNet169 exactly like model_handler.py
        base_model = timm.create_model('densenet169', pretrained=True, num_classes=0)
        
        # Freeze early layers initially
        for param in base_model.parameters():
            param.requires_grad = False
        
        # Unfreeze last N dense blocks
        if unfreeze_layers > 0:
            for block in list(base_model.features.children())[-unfreeze_layers:]:
                for param in block.parameters():
                    param.requires_grad = True
        
        # Custom head (matches model_handler.py)
        self.backbone = base_model
        num_features = base_model.num_features
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, NUM_CLASSES)
        )
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Only optimize unfrozen params + head
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
        return [optimizer], [scheduler]


def export_to_tf_format(model: DenseNetFractureModel, output_dir: str, img_size: int = 224):
    """Export to TensorFlow/Keras format compatible with model_handler.py"""
    import tensorflow as tf
    from tensorflow.keras.applications import DenseNet169
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
    from tensorflow.keras.models import Model
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Recreate exact model_handler.py architecture
    base_model = DenseNet169(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    
    # Freeze base
    for layer in base_model.layers:
        layer.trainable = False
    
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    
    tf_model = Model(inputs=base_model.input, outputs=predictions)
    
    # Transfer weights from PyTorch to TensorFlow
    model.eval()
    pt_state = model.state_dict()
    
    # Map PyTorch weights to TF layers
    # This is approximate - for exact transfer, train in TF directly
    print("Note: For exact weight transfer, consider training directly in TensorFlow")
    print("Saving PyTorch checkpoint for ONNX export instead...")
    
    # Save PyTorch checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': FRACTURE_TYPES,
        'img_size': img_size,
        'mean': IMAGENET_MEAN,
        'std': IMAGENET_STD,
    }, output_dir / "densenet169_fracture_finetuned.pt")
    
    print(f"Saved PyTorch checkpoint to {output_dir}/densenet169_fracture_finetuned.pt")


def export_to_onnx(model: DenseNetFractureModel, output_path: str, img_size: int = 224):
    """Export to ONNX for custom_model_handler.py"""
    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size)
    
    torch.onnx.export(
        model, dummy_input, output_path,
        export_params=True, opset_version=17,
        do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Exported ONNX to {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune DenseNet169 for fracture detection")
    parser.add_argument("--data-dir", default="data/fracture_classification", help="Dataset directory")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--unfreeze", type=int, default=3, help="Number of dense blocks to unfreeze")
    parser.add_argument("--fast-dev-run", action="store_true")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Data
    train_dataset = FractureDataset(args.data_dir, "train", get_transforms(args.img_size, True), args.img_size)
    val_dataset = FractureDataset(args.data_dir, "val", get_transforms(args.img_size, False), args.img_size)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # Model
    model = DenseNetFractureModel(lr=args.lr, unfreeze_layers=args.unfreeze)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(dirpath=output_dir/"checkpoints", filename="densenet169-{epoch:02d}-{val_acc:.4f}",
                       monitor='val_acc', mode='max', save_top_k=2),
        EarlyStopping(monitor='val_acc', patience=7, mode='max'),
    ]
    
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='auto',
        devices='auto',
        precision='16-mixed',
        callbacks=callbacks,
        fast_dev_run=args.fast_dev_run,
        gradient_clip_val=1.0,
    )
    
    trainer.fit(model, train_loader, val_loader)
    
    # Load best and export
    best_path = trainer.checkpoint_callback.best_model_path
    best_model = DenseNetFractureModel.load_from_checkpoint(best_path)
    
    # Export ONNX
    onnx_path = output_dir / "densenet169_fracture_finetuned.onnx"
    export_to_onnx(best_model, str(onnx_path), args.img_size)
    
    # Save class info
    class_info = {
        "classes": FRACTURE_TYPES,
        "class_to_idx": CLASS_TO_IDX,
        "img_size": args.img_size,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "architecture": "densenet169"
    }
    with open(output_dir / "class_info.json", "w") as f:
        json.dump(class_info, f, indent=2)
    
    print(f"\nDone! ONNX model: {onnx_path}")
    print(f"Use: USE_CUSTOM_MODEL=true python app.py")


if __name__ == "__main__":
    main()