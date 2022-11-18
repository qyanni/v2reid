from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# MODEL 
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# ID number of GPU
_C.MODEL.DEVICE_ID = '3'
# Name of the backbone (options: 'volo_d1', 'volo_d2', 'volo_d3', 'volo_d4', 'volo_d5')
_C.MODEL.NAME = 'volo_d1'
# Start with a pretrained version of the specified network (options: True/False)
_C.MODEL.PRETRAINED = False
# Path to pretrained model 
_C.MODEL.PRETRAIN_PATH = '/home/yan/Documents/thesis/volo_o/checkpoints/d1_224_84.2.pth.tar'
# Training with overlapping patches (options: True or False)
_C.MODEL.OVERLAP = False
# Training with batchnorm neck, layersnorm neck or none (options: 'bnneck', 'lnneck' or 'off')
_C.MODEL.NECK = 'off'
# Dropout rates 
_C.MODEL.DROP_PATH = 0.0
_C.MODEL.DROP_OUT = 0.0
_C.MODEL.ATT_DROP_RATE = 0.0

# -----------------------------------------------------------------------------
# LOSS
# -----------------------------------------------------------------------------
_C.LOSS = CN()
# Weight for ID loss
_C.LOSS.ID_LOSS_WEIGHT = 1.0 
# Training with triplet loss (options: True or False)
_C.LOSS.TRIPLET_LOSS = True
# Weight for Triplet loss
_C.LOSS.TRIPLET_LOSS_WEIGHT = 1.0 
# Margin of triplet loss
_C.LOSS.TRIPLET_LOSS_MARGIN = 0.3
# Training with center loss (options: True or False)
_C.LOSS.CENTER_LOSS = True
# Weight for center loss
_C.LOSS.CENTER_LOSS_WEIGHT = 0.0005

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Root directory where datasets are
_C.INPUT.ROOT_DIR = ('...')
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [224, 224]
# Size of the image during test
_C.INPUT.SIZE_TEST = [224, 224]
# Random probability for image horizontal flip
_C.INPUT.FLIP_PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# Values used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C.INPUT.PADDING = 10
# Crop a random portion of image and resize it to a given size
_C.INPUT.RESIZECROP = False
# change contrast, hue, etc
_C.INPUT.COLORJIT_PROB = 0.0
# value for distortion prob
_C.INPUT.DISTORTION_PROB = 0.0

# -----------------------------------------------------------------------------
# DATALOADER
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8 
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16
# Number of images per batch
_C.DATALOADER.IMS_PER_BATCH = 256

# ---------------------------------------------------------------------------- #
# SOLVER
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Seed
_C.SOLVER.SEED = 1234
# Optimizer (options: 'SGD', 'AdamW', etc)
_C.SOLVER.OPTIMIZER_NAME = "SGD"
# Whether using larger learning rate for fc layer
_C.SOLVER.LARGE_FC_LR = False
# Factor of learning bias
_C.SOLVER.BIAS_LR_FACTOR = 1
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.CENTER_LR = 0.5
# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005
# Max epochs
_C.SOLVER.MAX_EPOCHS = 300
# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 10
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 50

# Base learning rate
_C.SOLVER.BASE_LR = 3e-4
# type of scheduler (options: 'step' or 'cosine')
_C.SOLVER.SCHED = 'cosine'
#  warm up epochs
_C.SOLVER.WARMUP_EPOCHS = 10
# k decay as described in https://arxiv.org/abs/2004.05909
_C.SOLVER.K_DECAY = 1
 # the number of iterations (epochs) in the i-th decay cycle
_C.SOLVER.CYCLE_MUL = 1
# number of max restarts
_C.SOLVER.CYCLE_LIM = 1 
# decay rate
_C.SOLVER.DECAY_RATE = 0.1
#number of epochs for scheduler
_C.SOLVER.DECAY_T = 300
# If set to True, then every new epoch number equals epoch = epoch - warmup_t
_C.SOLVER.WARMUP_PRE = False

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = "myoutputs"
