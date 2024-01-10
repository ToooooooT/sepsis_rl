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
        done_indexs = np.where(self.dones == 1)[0]
        start_indexs = np.append(0, done_indexs[:-1] + 1)
        self.max_length = (done_indexs - start_indexs + 1).max()

    def estimate(self, **kwargs) -> Tuple[float, np.ndarray]:
        '''
        Description:
            compute all trajectory total reward and weight imporatance sampling.
        Args:
            policy_actions     : np.ndarray; expected shape (B, 1)
            policy_action_probs: np.ndarray; expected shape (B, D)
        Returns:
            avg_policy_return   : average policy return
            policy_return       : expected return of each patient; numpy array expected shape (1, B)
        '''
        policy_action_probs = kwargs['policy_action_probs']
        # \rho_t = \pi_1(a_t | s_t) / \pi_0(a_t | s_t), assume \pi_0(a_t | s_t) = 1
        rhos = policy_action_probs[np.arange(policy_action_probs.shape[0]), 
                                   self.actions.astype(np.int32).reshape(-1,)]
        # let the minimum probability be 0.01 to avoid nan
        rhos[rhos < 0.01] = 0.01

        # done and start index
        done_indexs = np.where(self.dones == 1)[0]
        start_indexs = np.append(0, done_indexs[:-1] + 1)

        n = done_indexs.shape[0]
        policy_return = np.zeros((n,), dtype=np.float64) 
        weights = np.zeros((n, self.max_length)) # the patient max length is 20 
        length = np.zeros((n,), dtype=np.int32) # the horizon length of each patient
        
        for i in range(n):
            start = start_indexs[i]
            end = done_indexs[i]
            total_reward = 0
            length[i] = end - start + 1
            for j, index in enumerate(range(end, start - 1, -1)):
                # let the minimum probability be 0.01 to avoid nan
                weights[i, end - start - j] = rhos[index]
                total_reward = self.gamma * total_reward + self.rewards[index]
            # \rho1:H * (\sum_{t=1}^H \gamma^{t-1} r_t) 
            policy_return[i] = np.cumprod(weights[i])[length[i] - 1] * total_reward

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
        done_indexs = np.where(self.dones == 1)[0]
        start_indexs = np.append(0, done_indexs[:-1] + 1)
        self.max_length = (done_indexs - start_indexs + 1).max()

    def estimate(self, **kwargs) -> Tuple[float, np.ndarray]:
        '''
        Description:
            compute all trajectory total reward and weight imporatance sampling.
        Args:
            policy_actions     : np.ndarray; expected shape (B, 1)
            policy_action_probs: np.ndarray; expected shape (B, D)
        Returns:
            avg_policy_return   : average policy return
            policy_return       : expected return of each patient; numpy array expected shape (1, B)
        '''
        policy_action_probs = kwargs['policy_action_probs']
        # \rho_t = \pi_1(a_t | s_t) / \pi_0(a_t | s_t), assume \pi_0(a_t | s_t) = 1
        rhos = policy_action_probs[np.arange(policy_action_probs.shape[0]), 
                                   self.actions.astype(np.int32).reshape(-1,)]
        # let the minimum probability be 0.01 to avoid nan
        rhos[rhos < 0.01] = 0.01

        # done and start index
        done_indexs = np.where(self.dones == 1)[0]
        start_indexs = np.append(0, done_indexs[:-1] + 1)

        n = done_indexs.shape[0]
        policy_return = np.zeros((n,), dtype=np.float64) 
        length = np.zeros((n,), dtype=np.int32) # the horizon length of each patient
        length2sumwight = np.zeros((self.max_length + 1,), dtype=np.float64)
        W_l = np.zeros((self.max_length + 1,), dtype=np.float64)
        
        for i in range(n):
            total_reward = 0
            l = done_indexs[i] - start_indexs[i] + 1
            w = 1
            for index in range(done_indexs[i], start_indexs[i] - 1, -1):
                w *= rhos[index]
                total_reward = self.gamma * total_reward + self.rewards[index]
            # \rho1:H * (\sum_{t=1}^H \gamma^{t-1} r_t) 
            policy_return[i] = w * total_reward
            length2sumwight[l] += w
            W_l[l] += 1
            length[i] = l

        for i in range(length.shape[0]):
            policy_return[i] *= W_l[length[i]] / length2sumwight[length[i]]

        policy_return = np.clip(policy_return, -self.clip_expected_return, self.clip_expected_return)
        return policy_return.mean(), policy_return.reshape(1, -1)
