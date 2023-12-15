import torch

class Config(object):
    def __init__(self):
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.EPISODE = 1e6

        # algorithm control
        self.USE_PRIORITY_REPLAY = False
        
        # Multi-step returns
        self.N_STEPS = 1

        # misc agent variables
        self.GAMMA = 0.99
        self.LR = 1e-4

        # memory
        self.TARGET_NET_UPDATE_FREQ = 1
        self.EXP_REPLAY_SIZE = 100000
        self.BATCH_SIZE = 32
        self.PRIORITY_ALPHA = 0.6
        self.PRIORITY_BETA_START = 0.4
        self.PRIORITY_BETA_FRAMES = 20000

        # data logging parameters
        self.ACTION_SELECTION_COUNT_FREQUENCY = 1000

        # Entropy regularization coefficient
        self.ALPHA = 0.2
        # Behavior cloning use in SAC
        self.BEHAVIOR_CLONING = True
        # Automatic entropy tuning
        self.AUTOTUNE = True
        self.TARGET_ENTROPY_SCALE = 0.89

        self.REG_LAMBDA = 5
        self.REWARD_THRESHOLD = 20
        self.IS_GRADIENT_CLIP = False

        # update target network parameter
        self.TAU = 0.005

        # coefficient of actor loss term in SAC + BC
        self.ACTOR_LAMBDA = 2.5

        self.SOFA_THRESHOLD = 4

        self.HIDDEN_SIZE = (128, 128)
