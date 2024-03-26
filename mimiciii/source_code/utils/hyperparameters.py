import torch
from typing import Dict

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
        self.Q_LR = 3e-4
        self.PI_LR = 3e-4

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

        # coefficient of actor loss term in BC
        self.ACTOR_LAMBDA = 2.5
        self.SOFA_THRESHOLD = 5
        self.BC_KL_BETA = 2e-1
        self.BC_TYPE = "cross_entropy"

        self.HIDDEN_SIZE = (128, 128)

        # data augmentation
        self.GAUSSIAN_NOISE_STD = 3e-4
        self.UNIFORM_NOISE = 3e-4
        self.MIXUP_ALPHA = 0.4
        self.ADVERSARIAL_STEP = 1e-4

        # CQL coefficient of regularization term in Q loss function
        self.ALPHA_PRIME = 1.0
        self.WITH_LAGRANGE = True
        self.TARGET_ACTION_GAP = 10.0

        self.USE_PI_B_EST = False
        self.USE_PI_B_KL = False

    def get_hyperparameters(self) -> Dict:
        config_dict = {key: value for key, value in vars(self).items() if not key.startswith('__')}
        return config_dict
