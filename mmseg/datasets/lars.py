import os
import os.path as osp
from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset

@DATASETS.register_module()
class LaRSDataset(BaseSegDataset):
    """
    LaRSDataset for semantic segmentation (3 classes: water, sky, obstacle).
    Directory structure expected:
        - img_dir: path to images
        - ann_dir: path to labels
        - Optional: split file listing image base names
    """

    METAINFO = dict(
        classes=('water', 'sky', 'obstacle'),
        palette=[
            (0, 0, 128),      # water: dark blue
            (128, 128, 128),  # sky: light gray
            (128, 0, 0),      # obstacle: maroon
        ]
    )

    def __init__(self,
                 img_dir,
                 ann_dir,
                 pipeline,
                 data_root=None,
                 split=None,
                 **kwargs):
        self.img_suffix = '.png'
        self.seg_map_suffix = '_label.png'
        super().__init__(
            img_dir=img_dir,
            ann_dir=ann_dir,
            pipeline=pipeline,
            data_root=data_root,
            split=split,
            **kwargs)


    def load_data_list(self):
        """Load image & annotation file paths."""

        if self.split is not None:
            with open(self.split, 'r') as f:
                lines = [x.strip() for x in f.readlines() if x.strip()]
            data_list = []
            for base_name in lines:
                data_info = dict(
                    img_path=osp.join(self.img_dir, base_name + self.img_suffix),
                    seg_map_path=osp.join(self.ann_dir, base_name + self.seg_map_suffix))
                data_list.append(data_info)
            return data_list

        # If no split file: scan all files in img_dir
        data_list = []
        for fname in os.listdir(self.img_dir):
            if fname.endswith(self.img_suffix):
                base_name = fname[:-len(self.img_suffix)]
                data_info = dict(
                    img_path=osp.join(self.img_dir, fname),
                    seg_map_path=osp.join(self.ann_dir, base_name + self.seg_map_suffix))
                data_list.append(data_info)
        return data_list
