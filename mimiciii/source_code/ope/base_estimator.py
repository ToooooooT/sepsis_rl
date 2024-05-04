from abc import ABC, abstractmethod
from typing import Dict, Tuple
import numpy as np

from utils import Config
from agents import BaseAgent

class BaseEstimator(ABC):
    def __init__(
        self, 
        agent: BaseAgent,
        data_dict: Dict, 
        config: Config,
        args
    ) -> None:
        super().__init__()
        '''
        Args:
            dataset    : unnormalization testing dataset, with action and reward 
            data_dict  : processed testing dataset; see preprocess/normalization.py
            args       : arguments from main file
        '''
        self.agent = agent
        self.states = data_dict['s']
        self.actions = data_dict['a']
        self.rewards = data_dict['r']
        self.next_states = data_dict['s_']
        self.dones = data_dict['done']
        self.clip_expected_return = args.clip_expected_return
        self.pi_b_est = config.USE_PI_B_EST
        self.gamma = config.GAMMA
        self.device = config.DEVICE
        self.done_indexs = np.where(self.dones == 1)[0]
        self.start_indexs = np.append(0, self.done_indexs[:-1] + 1)
        self.n = self.done_indexs.shape[0] # number of patients
        self.max_length = (self.done_indexs - self.start_indexs).max() + 1

    @abstractmethod
    def estimate(self, **kwargs) -> Tuple[float, np.ndarray]:
        ''' To override '''
        pass

    
    def get_rho(
        self, 
        policy_action_probs: np.ndarray, 
        behavior_action_probs: np.ndarray
    ) -> np.ndarray:
        '''
        Args:
            policy_action_probs  : np.ndarray; expected shape (B, D)
            behavior_action_probs: np.ndarray; expected shape (B, D)
        Returns:
            rhos : np.ndarray; expected shape (B,)
        '''
        # \rho_t = \pi_1(a_t | s_t) / \pi_0(a_t | s_t)
        if self.pi_b_est:
            # let the minimum probability of action be 0.01 to avoid nan
            behavior_action_probs[behavior_action_probs < 1e-2] = 1e-2
            policy_action_probs[policy_action_probs < 1e-2] = 1e-2

            rhos = policy_action_probs[np.arange(policy_action_probs.shape[0]), 
                                    self.actions.astype(np.int32).reshape(-1,)] / \
                    behavior_action_probs[np.arange(behavior_action_probs.shape[0]), 
                                    self.actions.astype(np.int32).reshape(-1,)]
        else:
            # assume \pi_0(a_t | s_t) = 1
            rhos = policy_action_probs[np.arange(policy_action_probs.shape[0]), 
                                    self.actions.astype(np.int32).reshape(-1,)]
            # let the minimum probability be 0.01 to avoid nan
            rhos[rhos < 1e-2] = 1e-2

        return rhos
