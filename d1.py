# Save this as debug_1.py in C:\Users\Payal\Desktop\mm-local

import sys
import os
sys.path.insert(0, os.getcwd())

from mmengine import Config

try:
    # Try to find the base config file
    base_config_paths = [
        './configs/mask2former/mask2former_r50_8xb2-90k_cityscapes-512x1024.py',
        './mask2former_r50_8xb2-90k_cityscapes-512x1024.py',
        'configs/mask2former/mask2former_r50_8xb2-90k_cityscapes-512x1024.py'
    ]
    
    base_cfg = None
    used_path = None
    
    for path in base_config_paths:
        if os.path.exists(path):
            print(f"Found base config at: {path}")
            base_cfg = Config.fromfile(path)
            used_path = path
            break
    
    if base_cfg is None:
        print("Base config file not found. Searching for mask2former configs...")
        for root, dirs, files in os.walk('.'):
            for file in files:
                if 'mask2former' in file and 'cityscapes' in file and file.endswith('.py'):
                    print(f"Found potential config: {os.path.join(root, file)}")
        exit(1)

    print("=== BASE CONFIG DECODE HEAD ===")
    print("Type:", base_cfg.model.decode_head.type)
    print("Keys:", list(base_cfg.model.decode_head.keys()))
    
    print("\n=== DECODE HEAD DETAILS ===")
    for key, value in base_cfg.model.decode_head.items():
        if key == 'pixel_decoder' and isinstance(value, dict):
            print(f"  {key}:")
            for pk, pv in value.items():
                print(f"    {pk}: {pv}")
        else:
            print(f"  {key}: {value}")

    print("\n=== SEARCHING FOR in_channels IN BASE CONFIG ===")
    def find_in_channels(d, path=""):
        results = []
        if isinstance(d, dict):
            for k, v in d.items():
                current_path = f"{path}.{k}" if path else k
                if k == "in_channels":
                    results.append((current_path, v))
                results.extend(find_in_channels(v, current_path))
        elif isinstance(d, list):
            for i, item in enumerate(d):
                results.extend(find_in_channels(item, f"{path}[{i}]"))
        return results

    in_channels_found = find_in_channels(base_cfg.model)
    for location, value in in_channels_found:
        print(f"Found in_channels at {location}: {value}")
        
    if not in_channels_found:
        print("No in_channels found in base configuration")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()