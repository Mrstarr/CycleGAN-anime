"""
Define some parameters in this script.
"""

# ----- MODIFY ME ----- #
START_EPOCH = 0  # epoch idx of beginning
NUM_EPOCH = 200  # all epoch
START_DECAY_EPOCH = 100  # epoch idx of decay
BATCH_SIZE = 1
DATA_ROOT = "datasets/Real2Anime2/"
LEARNING_RATE = 0.0002

# IMG_SIZE = 256
SIZE_H = 256
SIZE_W = 256
INPUT_CHANNEL = 3
OUTPUT_CHANNEL = 3
NUM_CPU = 2  # for dataloader

WEIGHT_IDENTITY = 5.0
WEIGHT_GAN = 1.0
WEIGHT_CYCLE = 10.0

# path
PATH_A2B = './output/<insert model name here>'
PATH_B2A = './output/<insert model name here>'

# ----- ENDING ----- #

# TODO: old params
# weights of mse
# WEIGHT_REG = 10
# WEIGHT_NOOBJ = 0.5
# WEIGHT_CLASS = 20


class Config:
    def __init__(self):
        # training
        self.start_epoch = START_EPOCH
        self.num_epoch = NUM_EPOCH
        self.start_decay_epoch = START_DECAY_EPOCH
        self.batch_size = BATCH_SIZE
        self.data_root = DATA_ROOT
        self.learning_rate = LEARNING_RATE
        # model
        self.w_identity = WEIGHT_IDENTITY
        self.w_gan = WEIGHT_GAN
        self.w_cycle = WEIGHT_CYCLE
        # image data
        self.size_h = SIZE_H
        self.size_w = SIZE_W
        self.input_ch = INPUT_CHANNEL
        self.output_ch = OUTPUT_CHANNEL
        # device
        self.num_cpu = NUM_CPU
        # path
        self.path_a2b = PATH_A2B
        self.path_b2a = PATH_B2A


# This global variable is for setting parameters
CONFIG = Config()
