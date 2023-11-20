import torch.nn as nn
from utils import Config
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, 
                 config: Config, 
                 env: dict, 
                 log_dir='./logs',
                 static_policy=False):
        # log directory
        self.log_dir = log_dir

        # environment
        self.num_feats = env['num_feats']
        self.num_actions = env['num_actions']

        # self.action_selections = [0 for _ in range(env['num_actions'])] # the frequency of each action be selected
        self.device = config.device

        # misc agent variables
        self.gamma = config.GAMMA
        self.lr = config.LR

        # update gradient
        self.is_gradient_clip = config.IS_GRADIENT_CLIP

        # algorithm control
        self.priority_replay = config.USE_PRIORITY_REPLAY

        # memory
        self.target_net_update_freq = config.TARGET_NET_UPDATE_FREQ
        self.experience_replay_size = config.EXP_REPLAY_SIZE
        self.batch_size = config.BATCH_SIZE
        self.priority_alpha = config.PRIORITY_ALPHA
        self.priority_beta_start = config.PRIORITY_BETA_START
        self.priority_beta_frames = config.PRIORITY_BETA_FRAMES

        # update target network parameter
        self.tau = config.TAU

        self.declare_memory()

        # network
        self.static_policy = static_policy

        self.declare_networks()


    @abstractmethod
    def save(self):
        ''' To override '''
        pass
    

    @abstractmethod
    def load(self):
        ''' To override '''
        pass

    @abstractmethod
    def declare_networks(self, t):
        ''' To override '''
        pass

    @abstractmethod
    def declare_memory(self, t):
        ''' To override '''
        pass

    @abstractmethod
    def update(self, t):
        ''' To override '''
        pass

    @abstractmethod
    def get_action(self, s, eps=0):
        ''' To override '''
        pass

    def update_target_model(self, target: nn.Module, source: nn.Module):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)




    # def save_action(self, actions):
    #     '''
    #     Args:
    #         actions: numpy.array; expected shape (B, 1)
    #     '''
    #     # save the frequency of each action be selected
    #     n = actions.shape[0]
    #     self.action_selections = [0 for _ in range(len(self.action_selections))]
    #     for action in actions:
    #         self.action_selections[int(action)] += (1.0 / n)