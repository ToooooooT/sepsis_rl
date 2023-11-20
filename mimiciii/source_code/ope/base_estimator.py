from abc import ABC, abstractmethod
import pandas as pd

from utils import Config

class BaseEstimator(ABC):
    def __init__(self, 
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
        self.dataset = dataset
        self.data = data_dict['data']
        self.id_index_map = data_dict['id_index_map']
        self.terminal_index = data_dict['terminal_index']
        self.clip_expected_return = args.clip_expected_return
        self.gamma = config.GAMMA

    @abstractmethod
    def estimate(self, **kwargs):
        ''' To override '''
        pass
