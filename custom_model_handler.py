#!/usr/bin/env python3
"""
Custom Fracture Detection Model Handler
Maintains EXACT same interface as model_handler.ModelHandler for drop-in replacement.
Uses ONNX Runtime for fast inference.

Configuration priority:
1. Constructor arguments
2. Environment variables (USE_CUSTOM_MODEL, CUSTOM_MODEL_PATH)
3. config.yaml
4. Defaults
"""

import os
import json
import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


FRACTURE_TYPES = {
    "transverse": {
        "description": "Clean break across the bone, typically caused by direct high-energy trauma.",
        "severity": "Moderate",
        "treatment": "1. Immobilization via splinting/casting\n2. Reduction if displaced\n3. Regular X-ray monitoring"
    },
    "oblique": {
        "description": "Angled break across the long axis of the bone, common in twisting injuries.",
        "severity": "Moderate to Severe",
        "treatment": "1. Orthopedic surgical evaluation\n2. Possible internal fixation (pins/plates)\n3. Targeted physical therapy"
    },
    "compound": {
        "description": "Open fracture where bone fragment pierces through the surrounding skin tissue.",
        "severity": "Severe (Medical Emergency)",
        "treatment": "1. Immediate emergency surgical intervention\n2. Broad-spectrum intravenous antibiotics\n3. Surgical debridement & wound care"
    },
    "stress": {
        "description": "Microscopic cracks in bone structure resulting from repetitive force or overuse.",
        "severity": "Mild to Moderate",
        "treatment": "1. Rest and strict non-weight-bearing\n2. Activity modification and supportive footwear\n3. Progressive rehabilitation"
    }
}

CLASS_NAMES = ["transverse", "oblique", "compound", "stress"]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class CustomModelHandler:
    """
    Drop-in replacement for model_handler.ModelHandler.
    Uses ONNX model trained on MURA dataset with modern architecture.
    
    Interface compatibility:
    - __init__(): No arguments required
    - load_model(): Loads ONNX model
    - predict(image_path) -> (fracture_key, confidence, info_dict)
    """
    
    def __init__(self, model_path: str = None, config_path: str = None):
        self.fracture_types = FRACTURE_TYPES
        self.class_names = CLASS_NAMES
        self.model = None
        self.session = None
        self.input_name = None
        self.output_name = None
        self.img_size = 224
        
        config = self._load_config(config_path)
        
        if model_path is None:
            model_path = config.get('model_path')
        
        if model_path is None:
            base_dir = Path(__file__).parent
            model_path = base_dir / "models" / "efficientnet_b4_fracture_classifier.onnx"
        
        self.model_path = str(model_path)
        self._class_info_path = str(Path(self.model_path).parent / "class_info.json")
        
        self._load_class_info()
    
    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration from config.yaml, environment variables, or defaults."""
        config = {}
        
        if config_path and os.path.exists(config_path):
            if HAS_YAML:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f) or {}
        
        if not config and HAS_YAML:
            default_config = Path(__file__).parent / "config.yaml"
            if default_config.exists():
                with open(default_config, 'r') as f:
                    config = yaml.safe_load(f) or {}
        
        env_model_path = os.getenv('CUSTOM_MODEL_PATH')
        if env_model_path:
            config['model_path'] = env_model_path
        
        return config

    def _load_class_info(self):
        """Load class mapping from training."""
        if os.path.exists(self._class_info_path):
            with open(self._class_info_path, 'r') as f:
                info = json.load(f)
                self.class_names = info.get("classes", CLASS_NAMES)
                self.img_size = info.get("img_size", 224)
                self.mean = np.array(info.get("mean", IMAGENET_MEAN.tolist()), dtype=np.float32)
                self.std = np.array(info.get("std", IMAGENET_STD.tolist()), dtype=np.float32)
        else:
            self.mean = IMAGENET_MEAN
            self.std = IMAGENET_STD

    def load_model(self):
        """Load ONNX model for inference."""
        if self.session is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"ONNX model not found at {self.model_path}. "
                    f"Train model first using train_fracture_model.py"
                )
            
            providers = ['CPUExecutionProvider']
            if ort.get_device() == 'GPU':
                providers.insert(0, 'CUDAExecutionProvider')
            
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            print(f"Loaded ONNX model from {self.model_path}")
            print(f"Input: {self.input_name}, Output: {self.output_name}")
        
        return self.session

    def _preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image exactly like training: resize, normalize, CHW format."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LANCZOS4)
        
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        
        return image.astype(np.float32)

    def predict(self, image_path: str) -> Tuple[str, float, Dict[str, Any]]:
        """
        Predict fracture type from image.
        
        Returns:
            Tuple of (fracture_key, confidence_percent, info_dict)
            Compatible with model_handler.ModelHandler.predict()
        """
        if self.session is None:
            self.load_model()
        
        input_tensor = self._preprocess_image(image_path)
        
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        logits = outputs[0][0]
        
        probs = self._softmax(logits)
        idx = int(np.argmax(probs))
        confidence = float(probs[idx] * 100)
        fracture_key = self.class_names[idx]
        
        return fracture_key, confidence, self.fracture_types[fracture_key]

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def predict_batch(self, image_paths: list) -> list:
        """Predict multiple images at once for efficiency."""
        if self.session is None:
            self.load_model()
        
        batch = np.stack([self._preprocess_image(p) for p in image_paths], axis=0)
        batch = np.squeeze(batch, axis=1)
        
        outputs = self.session.run([self.output_name], {self.input_name: batch})
        logits = outputs[0]
        
        results = []
        for i in range(len(image_paths)):
            probs = self._softmax(logits[i])
            idx = int(np.argmax(probs))
            confidence = float(probs[idx] * 100)
            fracture_key = self.class_names[idx]
            results.append((fracture_key, confidence, self.fracture_types[fracture_key]))
        
        return results


def get_model_handler(use_custom: bool = None) -> CustomModelHandler:
    """
    Factory function to get the appropriate model handler.
    
    Args:
        use_custom: If True, return CustomModelHandler. 
                    If False, return original ModelHandler.
                    If None, reads from USE_CUSTOM_MODEL env var.
    
    Usage in bone_detection_tab.py:
        from custom_model_handler import get_model_handler
        self.model_handler = get_model_handler()  # Auto-detects from env
    """
    if use_custom is None:
        use_custom = os.getenv('USE_CUSTOM_MODEL', 'false').lower() == 'true'
    
    if use_custom:
        return CustomModelHandler()
    else:
        from model_handler import ModelHandler
        return ModelHandler()


if __name__ == "__main__":
    import sys
    
    test_images = [
        "test_images/X-ray-1.png",
        "test_images/MRI-1.png",
        "test_images/CT-Scan-1.png",
    ]
    
    handler = CustomModelHandler()
    handler.load_model()
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\nTesting {img_path}...")
            fracture_type, confidence, info = handler.predict(img_path)
            print(f"  Type: {fracture_type}")
            print(f"  Confidence: {confidence:.2f}%")
            print(f"  Severity: {info['severity']}")
        else:
            print(f"Skipping {img_path} (not found)")