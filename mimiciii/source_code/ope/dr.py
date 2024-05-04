import numpy as np
from numpy import ndarray
import torch
import torch.nn as nn
from typing import Tuple

from ope.base_estimator import BaseEstimator
from utils import Config
from agents import BaseAgent, DQN, SAC

class DoublyRobust(BaseEstimator):
    def __init__(
        self, 
        agent: BaseAgent,
        data_dict: dict, 
        config: Config,
        args
    ) -> None:
        super().__init__(agent, data_dict, config, args)

    def estimate_values(self, q: nn.Module=None) -> Tuple[np.ndarray, np.ndarray]:
        '''
        param:
            q : q function from DM, e.g. fqe, ...
        '''
        states = torch.tensor(self.states, dtype=torch.float, device=self.device)
        with torch.no_grad():
            if q is None:
                if isinstance(self.agent, DQN):
                    self.agent.q.eval()
                    est_q_values, _ = self.agent.q(states).max(dim=1)
                    est_v_values = est_q_values
                elif isinstance(self.agent, SAC):
                    self.agent.q_dre.eval()
                    actions, _, _, action_probs = self.agent.get_action_probs(states)
                    q_values = self.agent.q_dre(states)
                    est_q_values = q_values.gather(1, actions)
                    est_v_values = (q_values * action_probs).sum(dim=1, keepdim=True)
            else:
                if isinstance(self.agent, DQN):
                    actions = self.agent.get_action_probs(states)[0]
                    est_q_values = q(states).gather(1, actions)
                    est_v_values = est_q_values
                else:
                    actions, _, _, action_probs = self.agent.get_action_probs(states)
                    q_values = q(states)
                    est_q_values = q_values.gather(1, actions)
                    est_v_values = (q_values * action_probs).sum(dim=1, keepdim=True)

        # (B, 1)
        return est_q_values.view(-1, 1).detach().cpu().numpy(),  \
                est_v_values.view(-1, 1).detach().cpu().numpy()
                 

    def estimate(self, **kwargs) -> Tuple[float, np.ndarray]:
        '''
        Args:
            policy_action_probs: np.ndarray; expected shape (B, D)
            behavior_action_probs: np.ndarray; expected shape (B, D)
        Returns:
            average policy return
            policy_return   : expected return of each patient; expected shape (1, B)
        '''
        self.agent = kwargs['agent']
        policy_action_probs = kwargs['policy_action_probs']
        behavior_action_probs = kwargs['behavior_action_probs']
        q = kwargs.get('q', None)
        est_q_values, est_v_values = self.estimate_values(q)

        rhos = self.get_rho(policy_action_probs, behavior_action_probs)

        policy_return = np.zeros((self.n,), dtype=np.float64) 

        for i in range(self.n):
            total_reward = 0
            gamma = 1
            prev_w = 1
            w = 1
            start_index, done_index = self.start_indexs[i], self.done_indexs[i] + 1
            
            rhos_i = rhos[start_index:done_index]
            rewards_i = self.rewards[start_index:done_index]
            est_v_values_i = est_v_values[start_index:done_index]
            est_q_values_i = est_q_values[start_index:done_index]

            for index in range(self.done_indexs[i] - self.start_indexs[i]):
                prev_w = w
                w *= rhos_i[index]
                total_reward += gamma * (w * rewards_i[index] + prev_w * est_v_values_i[index] \
                                         - w * est_q_values_i[index])
                gamma *= self.gamma
            policy_return[i] = total_reward

        policy_return = np.clip(policy_return, -self.clip_expected_return, self.clip_expected_return)
        return policy_return.mean(), policy_return.reshape(1, -1)


class PHWDR(DoublyRobust):
    def __init__(
        self, 
        agent: BaseAgent,
        data_dict: dict, 
        config: Config,
        args
    ) -> None:
        super().__init__(agent, data_dict, config, args)

    def estimate(self, **kwargs) -> Tuple[float, ndarray]:
        '''
        Args:
            policy_action_probs: np.ndarray; expected shape (B, D)
            behavior_action_probs: np.ndarray; expected shape (B, D)
        Returns:
            average policy return
            policy_return   : expected return of each patient; expected shape (1, B)
        '''
        self.agent = kwargs['agent']
        policy_action_probs = kwargs['policy_action_probs']
        behavior_action_probs = kwargs['behavior_action_probs']
        q = kwargs.get('q', None)
        est_q_values, est_v_values = self.estimate_values(q)

        rhos = self.get_rho(policy_action_probs, behavior_action_probs)

        policy_return = np.zeros((self.n,), dtype=np.float64) 
        length = np.zeros((self.n,), dtype=np.int32) # the horizon length of each patient
        ws = np.zeros((self.max_length + 1, self.max_length + 1), dtype=np.float64)
        W_l = np.zeros((self.max_length + 1,), dtype=np.float64)

        for i in range(self.n):
            start, end = self.start_indexs[i], self.done_indexs[i]
            l = end - start + 1
            ws[l, :l] += np.cumprod(rhos[start:end + 1])
            W_l[l] += 1

        for i in range(self.n):
            gamma = 1
            w = 1
            start_index, done_index = self.start_indexs[i], self.done_indexs[i] + 1 # 0 ~ 5
            l = done_index - start_index
            length[i] = l
            total_reward = est_v_values[start_index]
            
            rhos_i = rhos[start_index:done_index]
            rewards_i = self.rewards[start_index:done_index]
            est_v_values_i = est_v_values[start_index:done_index]
            est_q_values_i = est_q_values[start_index:done_index]
            est_v_values_i = np.append(est_v_values_i, 0)

            for index in range(l):
                total_reward += gamma * (w / ws[l, index]) * (rewards_i[index] + self.gamma * est_v_values_i[index + 1] \
                                             - est_q_values_i[index])
                w *= rhos_i[index]
                gamma *= self.gamma

            policy_return[i] = total_reward

        policy_return = np.clip(policy_return, -self.clip_expected_return, self.clip_expected_return)
        return policy_return.mean(), policy_return.reshape(1, -1)
