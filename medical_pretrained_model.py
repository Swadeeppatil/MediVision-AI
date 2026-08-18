#!/usr/bin/env python3
"""
Medical Pretrained Model for Fracture Detection
Uses CheXpert/MIMIC-CXR pretrained models from torchxrayvision
High accuracy without training from scratch.
"""

import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from pathlib import Path
import json
import timm



FRACTURE_TYPES = ["transverse", "oblique", "compound", "stress"]
NUM_CLASSES = len(FRACTURE_TYPES)
CLASS_TO_IDX = {c: i for i, c in enumerate(FRACTURE_TYPES)}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Medical pretrained model options
MEDICAL_MODELS = {
    "densenet121_imagenet": {
        "source": "timm",
        "model_name": "densenet121",
        "weights": "imagenet",
        "num_features": 1024,
    },
    "densenet169_imagenet": {
        "source": "timm",
        "model_name": "densenet169",
        "weights": "imagenet",
        "num_features": 1664,
    },
    "resnet50_imagenet": {
        "source": "timm",
        "model_name": "resnet50",
        "weights": "imagenet",
        "num_features": 2048,
    },
    "vit_base_patch16": {
        "source": "timm",
        "model_name": "vit_base_patch16_224",
        "weights": "imagenet",
        "num_features": 768,
    },
    "swin_base": {
        "source": "timm",
        "model_name": "swin_base_patch4_window7_224",
        "weights": "imagenet",
        "num_features": 1024,
    },
    "convnext_base": {
        "source": "timm",
        "model_name": "convnext_base",
        "weights": "imagenet",
        "num_features": 1024,
    },
    "efficientnet_b4": {
        "source": "timm",
        "model_name": "efficientnet_b4",
        "weights": "imagenet",
        "num_features": 1792,
    },
}


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


def get_transforms(img_size: int = 224, is_train: bool = True, model_source: str = "xrv"):
    """Get transforms appropriate for medical models."""
    if model_source == "xrv":
        # torchxrayvision expects specific normalization
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = IMAGENET_MEAN
        std = IMAGENET_STD
    
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.OneOf([
                A.GaussNoise(std=0.1),
                A.GaussianBlur(blur_limit=3),
            ], p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.CLAHE(clip_limit=2.0),
            ], p=0.3),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])


class MedicalFractureModel(pl.LightningModule):
    """Medical pretrained model with fracture classification head."""
    
    def __init__(
        self,
        model_key: str = "densenet121_chexpert",
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_key = model_key
        self.model_config = MEDICAL_MODELS[model_key]
        
        # Load medical pretrained backbone
        self.backbone, num_features = self._load_medical_backbone()
        
        # Freeze/unfreeze
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Classification head - handles both 2D (pooled) and 4D features
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1) if False else nn.Identity(),  # Will handle in forward
            nn.Flatten(start_dim=1),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, NUM_CLASSES)
        )
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.val_outputs = []

    def _load_medical_backbone(self):
        """Load pretrained backbone from timm."""
        config = self.model_config
        
        # Load from timm
        model = timm.create_model(config["model_name"], pretrained=True, num_classes=0)
        num_features = config["num_features"]
        print(f"Loaded pretrained: {config['model_name']} ({config['weights']})")
        return model, num_features

    def forward(self, x):
        features = self.backbone(x)
        # Handle different backbone outputs
        if features.dim() == 4:
            # CNN features: [batch, channels, H, W] -> [batch, channels]
            features = features.mean(dim=[2, 3])
        elif features.dim() == 3:
            # ViT/Swin features: [batch, seq_len, features] -> [batch, features]
            features = features.mean(dim=1)
        # features now: [batch, num_features]
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
        self.val_outputs.append({'preds': logits.argmax(dim=1), 'targets': y})
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        if not self.val_outputs:
            return
        all_preds = torch.cat([o['preds'] for o in self.val_outputs])
        all_targets = torch.cat([o['targets'] for o in self.val_outputs])
        
        for i, cls in enumerate(FRACTURE_TYPES):
            mask = all_targets == i
            if mask.sum() > 0:
                cls_acc = (all_preds[mask] == all_targets[mask]).float().mean()
                self.log(f'val_acc_{cls}', cls_acc)
        
        self.val_outputs.clear()

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)
        return [optimizer], [scheduler]


def export_to_onnx(model: MedicalFractureModel, output_path: str, img_size: int = 224):
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
    parser = argparse.ArgumentParser(description="Medical pretrained model for fracture detection")
    parser.add_argument("--data-dir", default="data/fracture_classification")
    parser.add_argument("--model", default="densenet121_chexpert", choices=list(MEDICAL_MODELS.keys()))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze pretrained backbone")
    parser.add_argument("--fast-dev-run", action="store_true")
    args = parser.parse_args()
    
    
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Data
    model_source = MEDICAL_MODELS[args.model]["source"]
    train_dataset = FractureDataset(args.data_dir, "train", 
                                     get_transforms(args.img_size, True, model_source), args.img_size)
    val_dataset = FractureDataset(args.data_dir, "val",
                                   get_transforms(args.img_size, False, model_source), args.img_size)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)
    
    # Model
    model = MedicalFractureModel(
        model_key=args.model,
        lr=args.lr,
        freeze_backbone=args.freeze_backbone,
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(dirpath=output_dir/"checkpoints", 
                       filename=f"{args.model}-{{epoch:02d}}-{{val_acc:.4f}}",
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
    
    # Export best model
    best_path = trainer.checkpoint_callback.best_model_path
    best_model = MedicalFractureModel.load_from_checkpoint(best_path)
    
    onnx_path = output_dir / f"{args.model}_fracture.onnx"
    export_to_onnx(best_model, str(onnx_path), args.img_size)
    
    # Save class info
    class_info = {
        "classes": FRACTURE_TYPES,
        "class_to_idx": CLASS_TO_IDX,
        "img_size": args.img_size,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "architecture": args.model,
        "source": model_source,
    }
    with open(output_dir / "class_info.json", "w") as f:
        json.dump(class_info, f, indent=2)
    
    print(f"\nDone! ONNX model: {onnx_path}")
    print(f"Use: USE_CUSTOM_MODEL=true python app.py")


if __name__ == "__main__":
    main()