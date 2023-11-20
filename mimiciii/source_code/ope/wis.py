import numpy as np
import pandas as pd

from ope.base_estimator import BaseEstimator
from utils import Config

class WIS(BaseEstimator):
    def __init__(self, 
                 dataset: pd.DataFrame, 
                 data_dict: dict, 
                 config: Config,
                 args) -> None:
        super().__init__(dataset, data_dict, config, args)


    def estimate(self, **kwargs):
        '''
        Description:
            compute all trajectory total reward and weight imporatance sampling.
        Args:
            action_probs    : policy action probabilities; numpy.ndarray expected shape (B, D)
        Returns:
            avg_policy_return   : average policy return
            policy_return       : expected return of each patient; numpy array expected shape (1, B)
        '''
        # 
        action_probs = kwargs['action_probs']
        num = len(self.id_index_map)
        policy_return = np.zeros((num,), dtype=np.float64) 
        weights = np.zeros((num, 50)) # assume the patient max length is 50 
        length = np.zeros((num,), dtype=np.int32) # the horizon length of each patient
        # \rho_t = \pi_1(a_t | s_t) / \pi_0(a_t | s_t), assume \pi_0(a_t | s_t) = 1
        rhos = action_probs[np.arange(action_probs.shape[0]), self.dataset.loc[:, 'action'].values]

        for i, id in enumerate(self.id_index_map.keys()):
            start, end = self.id_index_map[id][0], self.id_index_map[id][-1]
            assert(50 >= end - start + 1)
            reward = 0
            length[i] = int(end - start + 1)
            for j, index in enumerate(range(end, start - 1, -1)):
                # let the minimum probability be 0.01 to avoid nan
                weights[i, end - start - j] = max(rhos[index], 0.01)
                # total reward
                reward = self.gamma * reward + self.dataset.loc[index, 'reward']

            policy_return[i] = np.cumprod(weights[i])[length[i] - 1] * reward

        for i, l in enumerate(length):
            w_H = np.cumprod(weights[l <= length], axis=1)[:, l - 1].mean()
            policy_return[i] /= w_H

        policy_return = np.clip(policy_return, -self.clip_expected_return, self.clip_expected_return)
        return policy_return.mean(), policy_return.reshape(1, -1)