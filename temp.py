# import mmcv
# print(mmcv.__version__)


# from mmcv.ops import ms_deform_attn
# print(ms_deform_attn)


from mmseg.datasets.builder import DATASETS
print('Is LaRSDataset registered?', 'LaRSDataset' in DATASETS.module_dict)