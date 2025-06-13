default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,                      # Enables CUDNN auto-tuner to find the best algorithm
    mp_cfg=dict(mp_start_method='fork',        # Multi-processing method: 'fork' is default on Unix
                opencv_num_threads=0),         # Disable OpenCV's internal threading to avoid conflict
    dist_cfg=dict(backend='nccl'),             # Sets the backend for distributed training to 'nccl' (GPU-friendly)
                                               # to let gpu communicate efficiently with each other

)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

tta_model = dict(type='SegTTAModel')

'''
  TTA is a technique where:
  The input image is changed slightly (like flipped or resized).
  The model makes predictions on each version.
  Then all those predictions are combined (averaged) to get a final, better prediction.
'''