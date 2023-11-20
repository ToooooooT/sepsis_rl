from abc import ABC, abstractmethod
import pandas as pd

from utils import Config
from agents import BaseAgent

class BaseEstimator(ABC):
    def __init__(self, 
                 agent: BaseAgent,
                 dataset: pd.DataFrame, 
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
        self.dataset = dataset
        self.states = data_dict['data']['s']
        self.actions = data_dict['data']['a']
        self.rewards = data_dict['data']['r']
        self.next_states = data_dict['data']['s_']
        self.dones = data_dict['data']['done']
        self.id_index_map = data_dict['id_index_map']
        self.terminal_index = data_dict['terminal_index']
        self.clip_expected_return = args.clip_expected_return
        self.gamma = config.GAMMA
        self.device = config.DEVICE

    @abstractmethod
    def estimate(self, **kwargs):
        ''' To override '''
        pass
