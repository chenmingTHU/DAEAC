from yacs.config import CfgNode as CN


_C = CN()

_C.SYSTEM = CN()
# The number of workers reading the data into memory
_C.SYSTEM.NUM_WORKERS = 4
# The saving directory
_C.SYSTEM.SAVE_PATH = '/home/workspace/mingchen/ECG_UDA/exp'


_C.TRAIN = CN()

# The number of pre-training epochs
_C.TRAIN.PRE_TRAIN_EPOCHS = 300
# The number of training epochs for domain adaptation step
_C.TRAIN.EPOCHS = 600
# The size of mini-batch
_C.TRAIN.BATCH_SIZE = 256
# The initial learninig rate
_C.TRAIN.LR = 0.001
# The decay rate for learning rate decay
_C.TRAIN.DECAY_RATE = 0.99
# The decay step for learning rate decay
_C.TRAIN.DECAY_STEP = 200
# The marker of using ImbalanceSampler
_C.TRAIN.IMBALANCE_SAMPLE = True


_C.SETTING = CN()

# The random seed used in the experiment
_C.SETTING.SEED = -1

# The model/network name
_C.SETTING.NETWORK = 'ACNN'
# Training dataset
_C.SETTING.TRAIN_DATASET = 'mitdb'
# Testing/Adapation dataset
_C.SETTING.TEST_DATASET = 'mitdb'
# Loss function
_C.SETTING.LOSS = 'CBLoss'
# The distance metrics
_C.SETTING.DISTANCE = 'Cosine'
# The number of channels(leads) used in the experiment
_C.SETTING.LEAD = 2
# The scale of unlabeled training data
_C.SETTING.UDA_NUM = 300
# The input format
_C.SETTING.BEAT_NUM = 1
# The input length of each sample
_C.SETTING.FIXED_LEN = 200
# The marker of using norm-alignment
_C.SETTING.NORM_ALIGN = False
# The marker of using entire center loss
_C.SETTING.CENTER = False
# The marker of using inter_loss function
_C.SETTING.INTER_LOSS = False
# The marker of using intra_loss function
_C.SETTING.INTRA_LOSS = False
# The marker of centering loss
_C.SETTING.CLoss = True
# The marker of using data augmentation
_C.SETTING.AUGMENTATION = False
# Whether using BN in the ASPP module
_C.SETTING.ASPP_BN = True
# Whether using ACTIVATION in the ASPP module
_C.SETTING.ASPP_ACT = True
# Whether executing re-training procedure
_C.SETTING.RE_TRAIN = False
# The optimizer
_C.SETTING.OPTIMIZER = 'Adam'
# The setting of norm-alignment
_C.SETTING.ALIGN_SET = 'soft'
# The dilations of ASPP module in ACNN, default=(1, 6, 12, 18)
_C.SETTING.DILATIONS = (1, 6, 12, 18)
# The marker for applying incremental thresholds for obtaining pesudo labels
_C.SETTING.INCRE_THRS = False
# The activation function ASPP used
_C.SETTING.ACT = 'tanh'
_C.SETTING.F_ACT = 'tanh'
# The marker of using CORAL for domain adaptation
_C.SETTING.CORAL = False
# The marker of applying residual transfer on classifier
_C.SETTING.RESIDUAL = False
# The number of SE block in ATT module
_C.SETTING.BANK_NUM = 3

_C.PARAMETERS = CN()

# The weight of L2 regularizer of parameters
_C.PARAMETERS.W_L2 = 5e-4
# The weight of major classification loss
_C.PARAMETERS.W_CLS = 10.0
# The weight of BINARY classification loss (if any)
_C.PARAMETERS.W_BIN = 0.1
# The weight of consistency constraint (if any)
_C.PARAMETERS.W_CON = 1e-2
# The weight of CORAL
_C.PARAMETERS.W_CORAL = 1.0

# The weight of center loss for source data
_C.PARAMETERS.BETA1 = 1.0
# The weight of center loss for target data
_C.PARAMETERS.BETA2 = 1.0
# The weight of center loss for alignmentation
_C.PARAMETERS.BETA = 5.0
# The weight of PROTOTYPE aligmentation loss
_C.PARAMETERS.BETA_MMD = 1e-1
# The weight of inter_loss function
_C.PARAMETERS.BETA_INTER = 1e-1
# The weight of intra_loss function
_C.PARAMETERS.BETA_INTRA = 1e-1
# The weight for entire center loss (if any)
_C.PARAMETERS.BETA_C = 1.0
# The weight of norm-alignment
_C.PARAMETERS.W_NORM = 0.005

# The threshold for inter loss
_C.PARAMETERS.THR_M = 1.0
# The thresholds for obtaing pesudo labels for each category
_C.PARAMETERS.THRS = (0.9, 0.8, 0.85, 0.95)

# The number of emsembling teacher model for obtaining pesudo labels
_C.PARAMETERS.EMSEMBLE_NUM = 5
# The gap between two models
_C.PARAMETERS.EMSEMBLE_STEP = 5

# The learning rate for updating centers
_C.PARAMETERS.LR_C = 0.1
# The learning rate for updating centers of source domain
_C.PARAMETERS.LR_C_S = 0.1
# The learning rate for updating centers of target domain
_C.PARAMETERS.LR_C_T = 0.1

# The beta parameter of Class-Balance Loss
_C.PARAMETERS.BETA_CB = 0.999
# The radius for hard norm-alignment
_C.PARAMETERS.RADIUS = 15.0

# The weight for reconstruction loss (if applied)
_C.PARAMETERS.W_RECON = 0.01

# The keeping rate of Dropout layers in ACNN
_C.PARAMETERS.P = 0.0

# The paramter of EWLoss that controls the weights
_C.PARAMETERS.N = 75000

# The temperature of Cross-entropy loss
_C.PARAMETERS.T = 1


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
