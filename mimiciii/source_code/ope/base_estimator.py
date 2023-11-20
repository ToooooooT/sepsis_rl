from abc import ABC, abstractmethod
import pandas as pd

from utils import Config
from agents import BaseAgent

class BaseEstimator(ABC):
    def __init__(self, 
                 agent: BaseAgent,
                 data_dict: dict, 
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

    @abstractmethod
    def estimate(self, **kwargs):
        ''' To override '''
        pass
