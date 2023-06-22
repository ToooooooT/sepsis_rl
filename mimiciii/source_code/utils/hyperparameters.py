import torch

class Config(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.EPISODE = 100000

        # algorithm control
        self.USE_PRIORITY_REPLAY = False
        
        # Multi-step returns
        self.N_STEPS = 1

        # misc agent variables
        self.GAMMA = 0.99
        self.LR = 1e-4

        # memory
        self.TARGET_NET_UPDATE_FREQ = 1000
        self.EXP_REPLAY_SIZE = 100000
        self.BATCH_SIZE = 32
        self.PRIORITY_ALPHA = 0.6
        self.PRIORITY_BETA_START = 0.9
        self.PRIORITY_BETA_FRAMES = 30000

        # data logging parameters
        self.ACTION_SELECTION_COUNT_FREQUENCY = 1000

        self.REG_LAMBDA = 5