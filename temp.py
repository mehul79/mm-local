# Force MMSeg to import all dataset modules
import mmseg.datasets  # 👈 this triggers __init__.py

from mmseg.registry import DATASETS

print('Is LaRSDataset registered?', 'LaRSDataset' in DATASETS.module_dict)

print('\nAll registered datasets:')
for name in sorted(DATASETS.module_dict.keys()):
    print(f" - {name}")
