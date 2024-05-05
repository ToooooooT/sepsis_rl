import torch.nn as nn
import torch
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple

from utils import Config
from replay_buffer import ExperienceReplayMemory, PrioritizedReplayMemory
from network import DuellingMLP

class BaseAgent(ABC):
    def __init__(
        self, 
        env: dict, 
        config: Config, 
        log_dir: str = './logs',
        static_policy: bool = False
    ):
        # log directory
        self.log_dir = log_dir

        # environment
        self.num_feats = env['num_feats']
        self.num_actions = env['num_actions']

        # self.action_selections = [0 for _ in range(env['num_actions'])] # the frequency of each action be selected
        self.device = config.DEVICE

        # misc agent variables
        self.gamma = config.GAMMA
        self.q_lr = config.Q_LR
        self.pi_lr = config.PI_LR

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

        # data augmentation
        self.gaussian_noise_std = config.GAUSSIAN_NOISE_STD
        self.uniform_noise = config.UNIFORM_NOISE
        self.mixup_alpha = config.MIXUP_ALPHA
        self.adversarial_step = config.ADVERSARIAL_STEP
        self.use_state_augmentation = config.USE_STATE_AUGMENTATION
        self.state_augmentation_type = config.STATE_AUGMENTATION_TYPE
        self.state_augmentation_num = config.STATE_AUGMENTATION_NUM

        # update target network parameter
        self.tau = config.TAU

        self.declare_memory()

        # network
        self.static_policy = static_policy
        self.hidden_size = config.HIDDEN_SIZE
        self.declare_networks()


    @abstractmethod
    def save(self, name: str = 'model.pth'):
        ''' To override '''
        pass
    

    @abstractmethod
    def load(self, name: str = 'model.pth'):
        ''' To override '''
        pass

    @abstractmethod
    def save_checkpoint(self, epoch: int, name: str = 'checkpoint.pth'):
        ''' To override '''
        pass
    

    @abstractmethod
    def load_checkpoint(self, name: str = 'checkpoint.pth'):
        ''' To override '''
        pass

    def declare_networks(self):
        self.q = DuellingMLP(self.num_feats, self.num_actions, hidden_size=self.hidden_size).to(self.device)
        self.target_q = DuellingMLP(self.num_feats, self.num_actions, hidden_size=self.hidden_size).to(self.device)

    @abstractmethod
    def update(self, t: int):
        ''' To override '''
        pass

    @abstractmethod
    def get_action(self, s: np.ndarray, eps: int = 0):
        ''' To override '''
        pass

    @abstractmethod
    def get_action_probs(self, states: torch.Tensor):
        ''' To override '''
        pass

    def declare_memory(self):
        dims = (self.num_feats, 1, 1, self.num_feats, 1)
        self.memory = ExperienceReplayMemory(self.experience_replay_size, dims) \
                        if not self.priority_replay else \
                        PrioritizedReplayMemory(self.experience_replay_size, dims, self.priority_alpha, self.priority_beta_start, self.priority_beta_frames, self.device)


    def append_to_replay(
        self, 
        s: np.ndarray,
        a: np.ndarray,
        r: np.ndarray,
        s_: np.ndarray,
        done: np.ndarray
    ):
        self.memory.push((s, a, r, s_, done))


    def prep_minibatch(self) -> tuple[torch.Tensor, 
                                      torch.Tensor,
                                      torch.Tensor,
                                      torch.Tensor,
                                      torch.Tensor,
                                      list,
                                      torch.Tensor]:
        '''
        Returns:
            states: expected shape (B, S)
            actions: expected shape (B, D)
            rewards: expected shape (B, 1)
            next_states: expected shape (B, S)
            dones: expected shape (B, 1)
            indices: a list of index of replay buffer only for PER 
            weights: expected shape (B,); weight for each transition, only for PER
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

    def augmentation(
        self, 
        states: torch.Tensor, 
        next_states: torch.Tensor, 
        rewards: torch.Tensor, 
        dones: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Description:
            state augmentation only augment the features that are not used for computing reward,
            don't change SOFA and lactate which are the last two columns in the state
        '''
        states =  states.unsqueeze(1).repeat(1, self.state_augmentation_num, 1)
        next_states = next_states.unsqueeze(1).repeat(1, self.state_augmentation_num, 1)
        rewards =  rewards.unsqueeze(1).repeat(1, self.state_augmentation_num, 1)
        dones =  dones.unsqueeze(1).repeat(1, self.state_augmentation_num, 1)
        if self.state_augmentation_type == "Gaussian":
            states[:, :, :-2] += torch.randn(states[:, :, :-2].shape, device=self.device) * self.gaussian_noise_std
            next_states[:, :, :-2] += torch.randn(next_states[:, :, :-2].shape, device=self.device) * self.gaussian_noise_std
        elif self.state_augmentation_type == "Uniform":
            states[:, :, :-2] += torch.zeros(states[:, :, :-2].shape, device=self.device) \
                                                    .uniform_(-self.uniform_noise, self.uniform_noise)
            next_states[:, :, :-2] += torch.zeros(next_states[:, :, :-2].shape, device=self.device) \
                                                    .uniform_(-self.uniform_noise, self.uniform_noise)
        elif self.state_augmentation_type == "Mixup":
            # vector or scalar ?
            lmbda = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample().to(self.device)
            states[:, :, :-2] = lmbda * states[:, :, :-2] + (1 - lmbda) * next_states[:, :, :-2]
        elif self.state_augmentation_type == "Adversarial":
            states, next_states = self.adversarial_state_training(states, next_states, rewards, dones)
        else:
            raise NotImplementedError
        return states, next_states # (B, num, S)

    @abstractmethod
    def adversarial_state_training(
        self, 
        states: np.ndarray, 
        next_states: np.ndarray, 
        rewards: np.ndarray,
        dones: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        '''
        To override
        for data augmentation
        '''
        pass
