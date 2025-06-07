# Save this as debug_3.py in C:\Users\Payal\Desktop\mm-local
# Replace 'your_config.py' with your actual config file name

import sys
import os
sys.path.insert(0, os.getcwd())

from mmengine import Config

# Replace this with your actual config file name
YOUR_CONFIG_FILE = 'your_swin_config.py'  # CHANGE THIS TO YOUR ACTUAL FILE

try:
    if not os.path.exists(YOUR_CONFIG_FILE):
        print(f"Config file {YOUR_CONFIG_FILE} not found!")
        print("Available .py files in current directory:")
        for f in os.listdir('.'):
            if f.endswith('.py'):
                print(f"  {f}")
        exit(1)

    print(f"Loading config from: {YOUR_CONFIG_FILE}")
    cfg = Config.fromfile(YOUR_CONFIG_FILE)

    print("=== YOUR CONFIG DECODE HEAD ===")
    print("Type:", cfg.model.decode_head.get('type', 'NOT SET'))
    print("Keys:", list(cfg.model.decode_head.keys()))
    
    print("\n=== DECODE HEAD FULL CONFIG ===")
    for key, value in cfg.model.decode_head.items():
        if isinstance(value, dict) and len(str(value)) > 100:
            print(f"  {key}: <dict with {len(value)} keys>")
            if key == 'pixel_decoder':
                print(f"    pixel_decoder keys: {list(value.keys())}")
        else:
            print(f"  {key}: {value}")

    print("\n=== BACKBONE CONFIG ===")
    print("Backbone type:", cfg.model.backbone.get('type', 'NOT SET'))
    print("Backbone embed_dims:", cfg.model.backbone.get('embed_dims', 'NOT SET'))
    
    print("\n=== SEARCHING FOR in_channels IN YOUR CONFIG ===")
    def deep_search(obj, target_key, path=""):
        results = []
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                if key == target_key:
                    results.append((current_path, value))
                results.extend(deep_search(value, target_key, current_path))
        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                results.extend(deep_search(item, target_key, f"{path}[{i}]"))
        return results

    in_channels_locations = deep_search(cfg.model, 'in_channels')
    if in_channels_locations:
        for location, value in in_channels_locations:
            print(f"Found in_channels at {location}: {value}")
    else:
        print("No in_channels found in your configuration")

    print("\n=== TESTING CONFIG MERGE ===")
    try:
        # Try to build just the decode head config to see what happens
        from mmseg.registry import MODELS
        decode_head_cfg = cfg.model.decode_head.copy()
        print("Decode head config before build:")
        print(f"  Keys: {list(decode_head_cfg.keys())}")
        
        # This might fail, but will show us exactly what's wrong
        print("Attempting to build decode head...")
        # Don't actually build, just validate the config
        
    except Exception as e:
        print(f"Config validation error: {e}")

except Exception as e:
    print(f"Error loading config: {e}")
    import traceback
    traceback.print_exc()