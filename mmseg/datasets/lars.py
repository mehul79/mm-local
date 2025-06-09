# lars.py

import os.path as osp
import os
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmseg.utils import print_log

@DATASETS.register_module()
class LaRSDataset(CustomDataset):
    """
    LaRSDataset—semantic segmentation for water/sky/obstacle (3 classes).
    We expect:
      - img_dir: path to "images/<split>/"
      - ann_dir: path to "annotations/<split>/"
      - file_suffix: ".png"  for images
      - seg_map_suffix: "_label.png" for annotations

    You can adjust CLASSES and PALETTE if you use more fine-grained obstacle subclasses.
    """

    CLASSES = ('water', 'sky', 'obstacle')
    PALETTE = [
        (  0,   0, 128),  # water: dark blue 
        (128, 128, 128),  # sky: light gray
        (128,   0,   0),  # obstacle: maroon
    ]

    def __init__(self,
                 img_dir,
                 ann_dir,
                 pipeline,
                 data_root=None,
                 split=None,
                 **kwargs):
        """
        Args:
            img_dir (str): Directory of images, relative to data_root.
            ann_dir (str): Directory of annotation masks, relative to data_root.
            pipeline (list[dict]): Processing pipeline.
            data_root (str | None): Optional root directory for img_dir and ann_dir.
            split (str | None): A text file listing image IDs (one ID per line); or None.
        """

        # 1) store dataset-specific attributes
        # If you have train.txt / val.txt / test.txt listing filenames (without extensions),
        # you can pass split="path/to/train.txt". Otherwise, set split=None to scan directories directly.
        self.img_dir = img_dir
        self.ann_dir = ann_dir

        # 2) Each CustomDataset subclass must define:
        #    self.file_client_args, self.seg_map_suffix, self.img_suffix
        # Defaults:
        self.img_suffix = '.png'
        self.seg_map_suffix = '_label.png'

        # 3) Call super to do the rest of the setup (file list, class indexing, pipeline)
        super(LaRSDataset, self).__init__(
            img_dir=img_dir,
            ann_dir=ann_dir,
            pipeline=pipeline,
            data_root=data_root,
            split=split,
            **kwargs)

        # 4) (Optional) sanity check on the number of samples:
        print_log(f'Loaded LaRS dataset: {len(self)} images', logger='current')

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split):
        """Load image & annotation filepaths from a given directory or a split txt file.

        Returns a list of dicts: [{"img_info": {"filename": xxx}, "ann_info": {"seg_map": yyy}}, …].

        If split is None, it will scan img_dir for all files ending in img_suffix.
        Otherwise, split is a path to a txt file with one filename (without suffix) per line.
        """
        # If split is specified (e.g. "data_root/splits/train.txt"), read it:
        if split is not None:
            # split file contains base filenames, e.g. "0001", "0002", …
            with open(split, 'r') as f:
                lines = [x.strip() for x in f.readlines() if x.strip()]
            img_infos = []
            for base_name in lines:
                img_path = osp.join(img_dir, base_name + img_suffix)
                ann_path = osp.join(ann_dir, base_name + seg_map_suffix)
                img_infos.append(dict(img_info=dict(filename=img_path),
                                      ann_info=dict(seg_map=ann_path)))
            return img_infos

        # If no split file: scan img_dir for all images
        img_names = [
            x[:-len(img_suffix)] for x in os.listdir(img_dir)
            if x.endswith(img_suffix)
        ]
        img_infos = []
        for name in img_names:
            img_path = osp.join(img_dir, name + img_suffix)
            ann_path = osp.join(ann_dir, name + seg_map_suffix)
            img_infos.append(dict(img_info=dict(filename=img_path),
                                  ann_info=dict(seg_map=ann_path)))
        return img_infos
