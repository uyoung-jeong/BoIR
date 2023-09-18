import os

from yacs.config import CfgNode as CN

_C = CN()

_C.CFG_NAME = ''
_C.OUTPUT_DIR = ''
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.DIST_BACKEND = 'nccl'
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.VERBOSE = True
_C.DDP = False

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'BoIR'
_C.MODEL.DEVICE = 'cuda'
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ''
_C.MODEL.SYNC_BN = False
_C.MODEL.BACKBONE = CN(new_allowed=True)
_C.MODEL.BIAS_PROB = 0.01
_C.MODEL.NECK = CN()
_C.MODEL.NECK.IN_CHANNELS = 480
_C.MODEL.NECK.MODULES = CN(new_allowed=True) # enable module-specific config
_C.MODEL.CenterHead = CN()
_C.MODEL.CenterHead.IN_CHANNELS = 256
_C.MODEL.CenterHead.CHANNELS = 128
_C.MODEL.CenterHead.OUT_CHANNELS = 1
_C.MODEL.BUKHead = CN()
_C.MODEL.BUKHead.IN_CHANNELS = 256
_C.MODEL.BUKHead.CHANNELS = 128
_C.MODEL.BUKHead.OUT_CHANNELS = 17
_C.MODEL.EmbHead = CN()
_C.MODEL.EmbHead.IN_CHANNELS = 256
_C.MODEL.EmbHead.CHANNELS = 128
_C.MODEL.EmbHead.OUT_CHANNELS = 128
_C.MODEL.BboxHead = CN()
_C.MODEL.BboxHead.IN_CHANNELS = 256
_C.MODEL.BboxHead.CHANNELS = 128
_C.MODEL.BboxHead.OUT_CHANNELS = 4
_C.MODEL.InstKptHead = CN()
_C.MODEL.InstKptHead.IN_CHANNELS = 480
_C.MODEL.InstKptHead.CHANNELS = 32
_C.MODEL.InstKptHead.OUT_CHANNELS = 17

_C.LOSS = CN()
_C.LOSS.MULTI_HEATMAP_LOSS_WEIGHT = 1.0
_C.LOSS.SINGLE_HEATMAP_LOSS_WEIGHT = 1.0
_C.LOSS.BBOX_MASK_LOSS_WEIGHT = 0.25
_C.LOSS.BBOX_MASK_LOSS_BG_SAMPLE_LEVEL = 0 # 0: bg sampling per instance. 1: bg sampling per image
_C.LOSS.BBOX_LOSS_WEIGHT = 0.25
_C.LOSS.AE_BETA = 10.0
_C.LOSS.EMB_LOSS_WEIGHT = 1.0

# DATASET related params
_C.DATASET = CN()
_C.DATASET.MAX_INSTANCES = 100

# single dataset training/evaluation configs
_C.DATASET.ROOT = 'data'
_C.DATASET.DATASET = 'ochuman'
_C.DATASET.NUM_KEYPOINTS = 17
_C.DATASET.TRAIN = 'val'
_C.DATASET.TEST = 'test'
_C.DATASET.FILTER_IMAGE = False
_C.DATASET.SIGMA = 2.0
_C.DATASET.FLIP = 0.5
_C.DATASET.FLIP_INDEX = []

# training data augmentation
_C.DATASET.MAX_ROTATION = 30
_C.DATASET.MIN_SCALE = 0.75
_C.DATASET.MAX_SCALE = 1.25
_C.DATASET.SCALE_TYPE = 'short'
_C.DATASET.MAX_TRANSLATE = 40
_C.DATASET.INPUT_SIZE = 512
_C.DATASET.OUTPUT_SIZE = 128

# train
_C.TRAIN = CN()

_C.TRAIN.LR_SCHEDULER = 'cosine'
_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.001
_C.TRAIN.MIN_LR = 1e-6

_C.TRAIN.OPTIMIZER = 'adamw'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.025
_C.TRAIN.NESTEROV = False
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 140
_C.TRAIN.WARMUP_EPOCHS = 0
_C.TRAIN.WARMUP_START_VALUE = 1e-6

_C.TRAIN.RESUME = False
_C.TRAIN.CHECKPOINT = ''

_C.TRAIN.IMAGES_PER_GPU = 10
_C.TRAIN.SHUFFLE = True
_C.TRAIN.MAX_PROPOSALS = 4
_C.TRAIN.KEYPOINT_THRESHOLD = 0.1

_C.TRAIN.AMP = False
_C.TRAIN.CLIP_GRAD = 10.0
_C.TRAIN.TRANSFER_DATASET = False

# testing
_C.TEST = CN()
_C.TEST.FLIP_TEST = False
_C.TEST.IMAGES_PER_GPU = 1
_C.TEST.MODEL_FILE = ''
_C.TEST.OKS_SCORE = 0.7
_C.TEST.OKS_SIGMAS = [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89]

_C.TEST.MAX_PROPOSALS = 30
_C.TEST.KPT_MAX_PROPOSALS = 10
_C.TEST.KEYPOINT_THRESHOLD = 0.01
_C.TEST.BUK_THRESHOLD = 0.1
_C.TEST.CENTER_POOL_KERNEL = 3

_C.TEST.POOL_THRESHOLD1 = 300
_C.TEST.POOL_THRESHOLD2 = 200

_C.TEST.REMOVE_AUX_HEAD = True
_C.TEST.DEBUG = False

if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
