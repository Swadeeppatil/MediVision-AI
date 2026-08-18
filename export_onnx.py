#!/usr/bin/env python3
"""
ONNX Export and Verification Script
Exports trained PyTorch model to ONNX and verifies numerical parity
"""

import torch
import onnx
import onnxruntime as ort
import numpy as np
import timm
import json
from pathlib import Path
import argparse


def load_pytorch_model(checkpoint_path: str, model_name: str, num_classes: int = 4):
    """Load model from PyTorch Lightning checkpoint."""
    from train_fracture_model import FractureModel
    
    model = FractureModel.load_from_checkpoint(
        checkpoint_path,
        model_name=model_name,
        num_classes=num_classes,
    )
    model.eval()
    return model


def export_to_onnx(model: torch.nn.Module, output_path: str, img_size: int = 224, opset: int = 17):
    """Export model to ONNX with dynamic batch size."""
    dummy_input = torch.randn(1, 3, img_size, img_size)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        verbose=False
    )
    print(f"Exported to {output_path}")


def simplify_onnx(onnx_path: str):
    """Simplify ONNX model using onnx-simplifier if available."""
    try:
        import onnxsim
        model = onnx.load(onnx_path)
        model_simp, check = onnxsim.simplify(model)
        if check:
            onnx.save(model_simp, onnx_path)
            print("ONNX model simplified successfully")
        else:
            print("Simplification check failed, keeping original")
    except ImportError:
        print("onnx-simplifier not installed, skipping simplification")


def verify_numerical_parity(pytorch_model: torch.nn.Module, onnx_path: str, img_size: int = 224, num_tests: int = 10):
    """Verify PyTorch and ONNX outputs match."""
    pytorch_model.eval()
    
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    max_diff = 0.0
    mean_diff = 0.0
    
    for i in range(num_tests):
        dummy_input = torch.randn(1, 3, img_size, img_size)
        numpy_input = dummy_input.numpy()
        
        with torch.no_grad():
            pytorch_out = pytorch_model(dummy_input).numpy()
        
        onnx_out = session.run(None, {'input': numpy_input})[0]
        
        diff = np.abs(pytorch_out - onnx_out).max()
        max_diff = max(max_diff, diff)
        mean_diff += np.abs(pytorch_out - onnx_out).mean()
    
    mean_diff /= num_tests
    
    print(f"Numerical verification ({num_tests} tests):")
    print(f"  Max absolute difference: {max_diff:.6f}")
    print(f"  Mean absolute difference: {mean_diff:.6f}")
    
    if max_diff < 1e-4:
        print("  ✓ PASSED: Numerical parity within tolerance")
        return True
    else:
        print("  ✗ FAILED: Numerical difference exceeds tolerance")
        return False


def benchmark_onnx(onnx_path: str, img_size: int = 224, num_runs: int = 100):
    """Benchmark ONNX inference speed."""
    import time
    
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    dummy_input = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        session.run(None, {'input': dummy_input})
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        session.run(None, {'input': dummy_input})
        times.append(time.perf_counter() - start)
    
    avg_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000
    
    print(f"ONNX Inference Benchmark ({num_runs} runs):")
    print(f"  Average: {avg_time:.2f} ms")
    print(f"  Std: {std_time:.2f} ms")
    print(f"  FPS: {1000/avg_time:.1f}")
    
    return avg_time


def main():
    parser = argparse.ArgumentParser(description="Export and verify ONNX model")
    parser.add_argument("--checkpoint", required=True, help="Path to PyTorch Lightning checkpoint")
    parser.add_argument("--model-name", default="efficientnet_b4", help="Model architecture name")
    parser.add_argument("--output", default="models/fracture_classifier.onnx", help="Output ONNX path")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--verify", action="store_true", help="Verify numerical parity")
    parser.add_argument("--benchmark", action="store_true", help="Run inference benchmark")
    parser.add_argument("--simplify", action="store_true", help="Simplify ONNX model")
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Loading PyTorch model...")
    model = load_pytorch_model(args.checkpoint, args.model_name, args.num_classes)
    
    print("Exporting to ONNX...")
    export_to_onnx(model, str(output_path), args.img_size)
    
    if args.simplify:
        print("Simplifying ONNX...")
        simplify_onnx(str(output_path))
    
    # Verify ONNX model loads
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation passed")
    
    if args.verify:
        print("\nVerifying numerical parity...")
        verify_numerical_parity(model, str(output_path), args.img_size)
    
    if args.benchmark:
        print("\nBenchmarking...")
        benchmark_onnx(str(output_path), args.img_size)
    
    print(f"\nDone! ONNX model saved to: {output_path}")


if __name__ == "__main__":
    main()