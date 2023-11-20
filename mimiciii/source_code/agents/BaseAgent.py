import torch.nn as nn
import torch
from abc import ABC, abstractmethod

from utils import Config
from replay_buffer import ExperienceReplayMemory, PrioritizedReplayMemory

class BaseAgent(ABC):
    def __init__(self, 
                 env: dict, 
                 config: Config, 
                 log_dir='./logs',
                 static_policy=False):
        # log directory
        self.log_dir = log_dir

        # environment
        self.num_feats = env['num_feats']
        self.num_actions = env['num_actions']

        # self.action_selections = [0 for _ in range(env['num_actions'])] # the frequency of each action be selected
        self.device = config.DEVICE

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
    def update(self, t):
        ''' To override '''
        pass

    @abstractmethod
    def get_action(self, s, eps=0):
        ''' To override '''
        pass

    def declare_memory(self):
        dims = (self.num_feats, 1, 1, self.num_feats, 1)
        self.memory = ExperienceReplayMemory(self.experience_replay_size, dims) \
                        if not self.priority_replay else \
                        PrioritizedReplayMemory(self.experience_replay_size, dims, self.priority_alpha, self.priority_beta_start, self.priority_beta_frames, self.device)


    def append_to_replay(self, s, a, r, s_, done):
        self.memory.push((s, a, r, s_, done))


    def prep_minibatch(self):
        '''
        Returns:
            states: expected shape (B, S)
            actions: expected shape (B, D)
            rewards: expected shape (B, 1)
            next_states: expected shape (B, S)
            dones: expected shape (B, 1)
            indices: a list of index
            weights: expected shape (B,)
        '''
        # random transition batch is taken from replay memory
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)

        states = torch.tensor(states, device=self.device, dtype=torch.float)
        actions = torch.tensor(actions, device=self.device, dtype=torch.int64)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float)
        next_states = torch.tensor(next_states, device=self.device, dtype=torch.float)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_target_model(self, target: nn.Module, source: nn.Module):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
