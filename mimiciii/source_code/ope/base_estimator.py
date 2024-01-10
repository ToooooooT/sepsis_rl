from abc import ABC, abstractmethod
from typing import Dict
import numpy as np

from utils import Config
from agents import BaseAgent

class BaseEstimator(ABC):
    def __init__(self, 
                 agent: BaseAgent,
                 data_dict: Dict, 
                 config: Config,
                 args) -> None:
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
        self.gamma = config.GAMMA
        self.device = config.DEVICE
        self.done_indexs = np.where(self.dones == 1)[0]
        self.start_indexs = np.append(0, self.done_indexs[:-1] + 1)
        self.n = self.done_indexs.shape[0] # number of patients
        self.max_length = (self.done_indexs - self.start_indexs).max() + 1

    @abstractmethod
    def estimate(self, **kwargs):
        ''' To override '''
        pass
