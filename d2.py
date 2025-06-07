# Save this as debug_2.py in C:\Users\Payal\Desktop\mm-local

import sys
import os
sys.path.insert(0, os.getcwd())

try:
    from mmseg.models.decode_heads.mask2former_head import Mask2FormerHead
    import inspect
    
    print("=== Mask2FormerHead __init__ signature ===")
    sig = inspect.signature(Mask2FormerHead.__init__)
    print("Parameters:")
    for param_name, param in sig.parameters.items():
        print(f"  {param_name}: {param}")

    print("\n=== Mask2FormerHead MRO (Method Resolution Order) ===")
    for i, cls in enumerate(Mask2FormerHead.__mro__):
        print(f"{i}: {cls}")

    print("\n=== Check which parent class accepts in_channels ===")
    for cls in Mask2FormerHead.__mro__:
        if hasattr(cls, '__init__'):
            try:
                sig = inspect.signature(cls.__init__)
                if 'in_channels' in sig.parameters:
                    print(f"✓ Class {cls.__name__} accepts in_channels parameter")
                    print(f"  Parameter: {sig.parameters['in_channels']}")
                else:
                    print(f"✗ Class {cls.__name__} does NOT accept in_channels")
            except Exception as e:
                print(f"? Could not check {cls.__name__}: {e}")

    print("\n=== Check MMSeg version ===")
    import mmseg
    print(f"MMSegmentation version: {mmseg.__version__}")
    
    print("\n=== Check MMEngine version ===")
    import mmengine
    print(f"MMEngine version: {mmengine.__version__}")

except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're in the correct environment and MMSegmentation is installed")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()