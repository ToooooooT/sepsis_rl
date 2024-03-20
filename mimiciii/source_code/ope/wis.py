import numpy as np
from typing import Tuple

from ope.base_estimator import BaseEstimator
from utils import Config
from agents import BaseAgent

class WIS(BaseEstimator):
    def __init__(self, 
                 agent: BaseAgent,
                 data_dict: dict, 
                 config: Config,
                 args) -> None:
        super().__init__(agent, data_dict, config, args)

    def estimate(self, **kwargs) -> Tuple[float, np.ndarray]:
        '''
        Description:
            compute all trajectory total reward and weight imporatance sampling.
        Args:
            policy_actions       : np.ndarray; expected shape (B, 1)
            policy_action_probs  : np.ndarray; expected shape (B, D)
            behavior_action_probs: np.ndarray; expected shape (B, D)
        Returns:
            avg_policy_return   : average policy return
            policy_return       : expected return of each patient; numpy array expected shape (1, B)
        '''
        policy_action_probs = kwargs['policy_action_probs']
        behavior_action_probs = kwargs['behavior_action_probs']

        rhos = self.get_rho(policy_action_probs, behavior_action_probs)

        policy_return = np.zeros((self.n,), dtype=np.float64) 
        weights = np.zeros((self.n, self.max_length)) # the patient max length is 20 
        length = np.zeros((self.n,), dtype=np.int32) # the horizon length of each patient
        
        for i in range(self.n):
            start, end = self.start_indexs[i], self.done_indexs[i]
            total_reward = 0
            length[i] = end - start + 1
            if start > 0:
                weights[i, :length[i]] = rhos[end:start - 1:-1]
            else:
                weights[i, :length[i]] = rhos[end::-1]
            total_reward = np.dot(self.rewards[start:end + 1, 0], self.gamma ** np.arange(length[i]))
            # \rho1:H * (\sum_{t=1}^H \gamma^{t-1} r_t) 
            policy_return[i] = np.cumprod(weights[i, :length[i]])[-1] * total_reward

        for i, l in enumerate(length):
            w_H = np.cumprod(weights[l <= length], axis=1)[:, l - 1].mean()
            policy_return[i] /= w_H

        policy_return = np.clip(policy_return, -self.clip_expected_return, self.clip_expected_return)
        return policy_return.mean(), policy_return.reshape(1, -1)


class PHWIS(BaseEstimator):
    def __init__(self, 
                 agent: BaseAgent,
                 data_dict: dict, 
                 config: Config,
                 args) -> None:
        super().__init__(agent, data_dict, config, args)

    def estimate(self, **kwargs) -> Tuple[float, np.ndarray]:
        '''
        Description:
            compute all trajectory total reward and weight imporatance sampling.
        Args:
            policy_actions       : np.ndarray; expected shape (B, 1)
            policy_action_probs  : np.ndarray; expected shape (B, D)
            behavior_action_probs: np.ndarray; expected shape (B, D)
        Returns:
            avg_policy_return   : average policy return
            policy_return       : expected return of each patient; numpy array expected shape (1, B)
        '''
        policy_action_probs = kwargs['policy_action_probs']
        behavior_action_probs = kwargs['behavior_action_probs']

        rhos = self.get_rho(policy_action_probs, behavior_action_probs)

        policy_return = np.zeros((self.n,), dtype=np.float64) 
        length = np.zeros((self.n,), dtype=np.int32) # the horizon length of each patient
        length2sumwight = np.zeros((self.max_length + 1,), dtype=np.float64)
        W_l = np.zeros((self.max_length + 1,), dtype=np.float64)
        
        for i in range(self.n):
            start, end = self.start_indexs[i], self.done_indexs[i]
            total_reward = 0
            l = end - start + 1
            w = np.prod(rhos[start:end + 1])
            total_reward = np.dot(self.rewards[start:end + 1, 0], self.gamma ** np.arange(l))
            # \rho1:H * (\sum_{t=1}^H \gamma^{t-1} r_t) 
            policy_return[i] = w * total_reward
            length2sumwight[l] += w
            W_l[l] += 1
            length[i] = l

        for i in range(self.n):
            policy_return[i] *= W_l[length[i]] / length2sumwight[length[i]]

        policy_return = np.clip(policy_return, -self.clip_expected_return, self.clip_expected_return)
        return policy_return.mean(), policy_return.reshape(1, -1)
