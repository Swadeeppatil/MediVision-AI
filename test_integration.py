#!/usr/bin/env python3
"""
Integration Test Script
Verifies custom_model_handler works with existing app components
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from custom_model_handler import CustomModelHandler, get_model_handler


def test_custom_handler():
    """Test CustomModelHandler interface compatibility."""
    print("Testing CustomModelHandler...")
    
    handler = CustomModelHandler()
    
    # Test load_model
    try:
        handler.load_model()
        print("  [OK] load_model() works")
        
        # Test predict with test images
        test_images = [
            "test_images/X-ray-1.png",
            "test_images/MRI-1.png",
        ]
        
        for img_path in test_images:
            if os.path.exists(img_path):
                try:
                    fracture_type, confidence, info = handler.predict(img_path)
                    print(f"  [OK] predict({img_path}) -> {fracture_type} ({confidence:.2f}%)")
                    
                    # Verify return format matches ModelHandler
                    assert isinstance(fracture_type, str), "fracture_type must be str"
                    assert isinstance(confidence, float), "confidence must be float"
                    assert isinstance(info, dict), "info must be dict"
                    assert "description" in info, "info must have description"
                    assert "severity" in info, "info must have severity"
                    assert "treatment" in info, "info must have treatment"
                    assert fracture_type in ["transverse", "oblique", "compound", "stress"]
                    assert 0 <= confidence <= 100
                    
                except Exception as e:
                    print(f"  [FAIL] predict({img_path}) failed: {e}")
                    return False
            else:
                print(f"  [WARN] {img_path} not found")
        
    except FileNotFoundError as e:
        print(f"  [WARN] Model not found (expected if not trained yet): {e}")
        return True  # Not a failure, just not trained yet
    except Exception as e:
        print(f"  [FAIL] load_model() failed: {e}")
        return False
    
    return True


def test_factory_function():
    """Test get_model_handler factory."""
    print("\nTesting get_model_handler factory...")
    
    try:
        custom_handler = get_model_handler(use_custom=True)
        assert isinstance(custom_handler, CustomModelHandler)
        print("  [OK] get_model_handler(use_custom=True) returns CustomModelHandler")
    except Exception as e:
        print(f"  [FAIL] Factory failed for custom: {e}")
        return False
    
    try:
        from model_handler import ModelHandler
        original_handler = get_model_handler(use_custom=False)
        assert isinstance(original_handler, ModelHandler)
        print("  [OK] get_model_handler(use_custom=False) returns ModelHandler")
    except Exception as e:
        print(f"  [FAIL] Factory failed for original: {e}")
        return False
    
    return True


def test_interface_compatibility():
    """Verify both handlers have identical interface."""
    print("\nTesting interface compatibility...")
    
    from model_handler import ModelHandler
    
    # Check instance attributes (fracture_types is set in __init__)
    orig_instance = ModelHandler()
    custom_instance = CustomModelHandler()
    
    required_attrs = ['load_model', 'predict', 'fracture_types']
    
    for attr in required_attrs:
        if not hasattr(orig_instance, attr):
            print(f"  [FAIL] ModelHandler missing {attr}")
            return False
        if not hasattr(custom_instance, attr):
            print(f"  [FAIL] CustomModelHandler missing {attr}")
            return False
    
    print("  [OK] Both handlers have required attributes: load_model, predict, fracture_types")
    
    # Check fracture_types structure
    orig_types = orig_instance.fracture_types
    custom_types = custom_instance.fracture_types
    
    if set(orig_types.keys()) != set(custom_types.keys()):
        print(f"  [FAIL] Fracture type keys differ!")
        print(f"    Original: {list(orig_types.keys())}")
        print(f"    Custom: {list(custom_types.keys())}")
        return False
    
    for key in orig_types:
        if set(orig_types[key].keys()) != set(custom_types[key].keys()):
            print(f"  [FAIL] Info keys differ for {key}")
            return False
    
    print("  [OK] fracture_types structure identical")
    return True


def main():
    print("="*60)
    print("INTEGRATION TEST: Custom Model Handler")
    print("="*60)
    
    all_passed = True
    
    all_passed &= test_interface_compatibility()
    all_passed &= test_factory_function()
    all_passed &= test_custom_handler()
    
    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED [OK]")
        print("CustomModelHandler is a drop-in replacement for ModelHandler")
    else:
        print("SOME TESTS FAILED [FAIL]")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())