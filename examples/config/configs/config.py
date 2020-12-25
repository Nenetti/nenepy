# ==================================================================================================
#
#   Config
#
# ==================================================================================================
from nenepy.utils.configs import Config
from nenepy.utils.dictionary import AttrDict

_C = Config()

# ==================================================================================================
#
#   General options
#
# ==================================================================================================
_C.NUM_GPUS = 1
_C.DEVICE = "cuda"
_C.LOG_DIR = "logs"
_C.TRIAL = "trial_01"
_C.PRETRAIN_DIR = "pretrain"
_C.WEIGHTS = "model.pth"
_C.LOG_FILE = "Log.yaml"
_C.LOG_CONFIG_FILE = "Config_Log.yaml"
_C.SAVE_INTERVAL = 5
_C.VALIDATE_INTERVAL = 20

# ==================================================================================================
#
#   DataSet options
#
# ==================================================================================================
_C.DATASET = AttrDict()
_C.DATASET.USE_GT_MASKS = False

_C.DATASET.TRAIN = AttrDict()
_C.DATASET.TRAIN.CROP_SIZE = [320, 320]
_C.DATASET.TRAIN.SCALE_FROM = 1.0
_C.DATASET.TRAIN.SCALE_TO = 1.0
_C.DATASET.TRAIN.APPLY_TRANSFORM = True

_C.DATASET.VALIDATE = AttrDict()
_C.DATASET.VALIDATE.CROP_SIZE = [320, 320]
_C.DATASET.VALIDATE.SCALE_FROM = 1.0
_C.DATASET.VALIDATE.SCALE_TO = 1.0
_C.DATASET.VALIDATE.APPLY_TRANSFORM = False

# ==================================================================================================
#
#   Training options
#
# ==================================================================================================
_C.TRAIN = AttrDict()
_C.TRAIN.BATCH_SIZE = 20
_C.TRAIN.PRETRAIN_BATCH_SIZE = 20
_C.TRAIN.NUM_EPOCHS = 15
_C.TRAIN.NUM_WORKERS = 4
_C.TRAIN.NUM_PRETRAIN = 20
_C.TRAIN.CONTINUE = False

# ==================================================================================================
#
#   Validate options
#
# ==================================================================================================
_C.VALIDATE = AttrDict()
_C.VALIDATE.BATCH_SIZE = 20
_C.VALIDATE.NUM_WORKERS = 4
_C.VALIDATE.CRF_PROCESS = False

# ==================================================================================================
#
#   Inference options
#
# ==================================================================================================
_C.TEST = AttrDict()
_C.TEST.BATCH_SIZE = 8
_C.TEST.CRF_PROCESS = False

# ==================================================================================================
#
#   Mask options
#
# ==================================================================================================
_C.MASK = AttrDict()
_C.MASK.CRF = AttrDict()
_C.MASK.CRF.ITERATION = 5

# ==================================================================================================
#
#   Model options
#
# ==================================================================================================
_C.NET = AttrDict()
_C.NET.LR = 0.001
_C.NET.WEIGHT_DECAY = 1e-5
_C.NET.MASK_LOSS_BCE = 1.0
_C.NET.FOCAL_P = 3
_C.NET.FOCAL_LAMBDA = 0.01
_C.NET.PAMR_KERNEL = [1, 2, 4, 8, 12, 24]
_C.NET.PAMR_ITER = 10
_C.NET.SG_PSI = 0.3
_C.NET.LOWER_LIMIT = 0.5
_C.NET.BACKBONE_PRETRAINED = True
