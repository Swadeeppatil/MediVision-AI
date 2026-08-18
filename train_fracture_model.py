#!/usr/bin/env python3
"""
Fracture Classification Training Script
Uses EfficientNet-B4 / ViT / Swin Transformer with PyTorch Lightning
Exports to ONNX for inference compatibility with existing app
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import json


TARGET_CLASSES = ["transverse", "oblique", "compound", "stress"]
NUM_CLASSES = len(TARGET_CLASSES)
CLASS_TO_IDX = {cls: i for i, cls in enumerate(TARGET_CLASSES)}
IDX_TO_CLASS = {i: cls for i, cls in enumerate(TARGET_CLASSES)}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FractureDataset(Dataset):
    def __init__(self, root_dir: str, split: str, transform=None, img_size: int = 224):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.img_size = img_size
        self.samples = []
        
        for class_name in TARGET_CLASSES:
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*.png"):
                    self.samples.append((str(img_path), CLASS_TO_IDX[class_name]))
        
        print(f"{split}: {len(self.samples)} samples")
        for cls in TARGET_CLASSES:
            count = sum(1 for _, idx in self.samples if idx == CLASS_TO_IDX[cls])
            print(f"  {cls}: {count}")

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
            A.VerticalFlip(p=0.1),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50)),
                A.GaussianBlur(blur_limit=3),
                A.MotionBlur(blur_limit=3),
            ], p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
                A.RandomGamma(gamma_limit=(80, 120)),
            ], p=0.3),
            A.ElasticTransform(alpha=1, sigma=5, alpha_affine=5, p=0.2),
            A.GridDistortion(p=0.2),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])


class FractureDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, img_size: int = 224, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = FractureDataset(
            self.data_dir, "train", 
            transform=get_transforms(self.img_size, is_train=True),
            img_size=self.img_size
        )
        self.val_dataset = FractureDataset(
            self.data_dir, "val",
            transform=get_transforms(self.img_size, is_train=False),
            img_size=self.img_size
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, persistent_workers=True
        )


class FractureModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "efficientnet_b4",
        num_classes: int = NUM_CLASSES,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        label_smoothing: float = 0.1,
        pretrained: bool = True,
        img_size: int = 224,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=num_classes,
            drop_rate=0.3,
            drop_path_rate=0.2,
        )
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.validation_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.validation_outputs.append({
            'preds': preds.detach().cpu(),
            'targets': y.detach().cpu(),
            'logits': logits.detach().cpu()
        })
        
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        if not self.validation_outputs:
            return
            
        all_preds = torch.cat([o['preds'] for o in self.validation_outputs])
        all_targets = torch.cat([o['targets'] for o in self.validation_outputs])
        all_logits = torch.cat([o['logits'] for o in self.validation_outputs])
        
        for i, cls in enumerate(TARGET_CLASSES):
            mask = all_targets == i
            if mask.sum() > 0:
                cls_acc = (all_preds[mask] == all_targets[mask]).float().mean()
                self.log(f'val_acc_{cls}', cls_acc)
        
        self.validation_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }


def export_to_onnx(model: FractureModel, output_path: str, img_size: int = 224):
    """Export PyTorch model to ONNX format."""
    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model exported to {output_path}")


def verify_onnx_model(onnx_path: str, img_size: int = 224):
    """Verify ONNX model runs correctly."""
    import onnxruntime as ort
    
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    dummy_input = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
    
    outputs = session.run(None, {'input': dummy_input})
    print(f"ONNX verification: output shape = {outputs[0].shape}")
    return outputs[0]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train fracture classification model")
    parser.add_argument("--data-dir", default="data/fracture_classification", help="Dataset directory")
    parser.add_argument("--model-name", default="efficientnet_b4", choices=[
        "efficientnet_b4", "efficientnet_b5", "vit_base_patch16_224", 
        "swin_base_patch4_window7_224", "convnext_base"
    ])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    parser.add_argument("--fast-dev-run", action="store_true", help="Quick test run")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_module = FractureDataModule(
        args.data_dir, args.batch_size, args.img_size, num_workers=4
    )
    
    model = FractureModel(
        model_name=args.model_name,
        lr=args.lr,
        img_size=args.img_size,
    )
    
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename=f"{args.model_name}-{{epoch:02d}}-{{val_acc:.4f}}",
            monitor='val_acc',
            mode='max',
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(monitor='val_acc', patience=10, mode='max'),
        LearningRateMonitor(logging_interval='epoch'),
    ]
    
    logger = TensorBoardLogger(
        save_dir=output_dir / "logs",
        name=args.model_name,
    )
    
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='auto',
        devices='auto',
        precision='16-mixed',
        callbacks=callbacks,
        logger=logger,
        fast_dev_run=args.fast_dev_run,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
    )
    
    if args.resume:
        trainer.fit(model, datamodule=data_module, ckpt_path=args.resume)
    else:
        trainer.fit(model, datamodule=data_module)
    
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f"Best model: {best_model_path}")
    
    best_model = FractureModel.load_from_checkpoint(best_model_path)
    
    onnx_path = output_dir / f"{args.model_name}_fracture_classifier.onnx"
    export_to_onnx(best_model, str(onnx_path), args.img_size)
    
    verify_onnx_model(str(onnx_path), args.img_size)
    
    class_info = {
        "classes": TARGET_CLASSES,
        "class_to_idx": CLASS_TO_IDX,
        "idx_to_class": IDX_TO_CLASS,
        "img_size": args.img_size,
        "model_name": args.model_name,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    }
    with open(output_dir / "class_info.json", "w") as f:
        json.dump(class_info, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"ONNX model: {onnx_path}")
    print(f"Class info: {output_dir / 'class_info.json'}")


if __name__ == "__main__":
    main()