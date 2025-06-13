from mmseg.datasets import build_dataset
from mmseg.datasets import builder
import os

# 1. Define a minimal config dict that matches our LaRS base
cfg = dict(
    type='LaRSDataset',
    data_root='/path/to/LaRS',       # <<–– change to your actual path
    img_dir='images/train',
    ann_dir='annotations/train',
    pipeline=[  # minimal pipeline: load only
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Normalize',
             mean=[123.675, 116.28, 103.53],
             std=[58.395, 57.12, 57.375],
             to_rgb=True),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg'])
    ]
)

# 2. Build dataset and fetch a few samples
dataset = build_dataset(cfg)

print(f'Total training images: {len(dataset)}')
for i in range(min(5, len(dataset))):
    data_sample = dataset[i]
    img = data_sample['img'].data[0]
    gt_seg = data_sample['gt_semantic_seg'].data[0]
    print(f'Sample {i}: image shape = {img.shape}, mask unique labels = {set(gt_seg.flatten().tolist())}')

