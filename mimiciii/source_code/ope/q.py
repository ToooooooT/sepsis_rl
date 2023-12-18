import numpy as np
import torch
from typing import Tuple

from utils import Config
from ope.base_estimator import BaseEstimator
from agents import DQN, SAC

class QEstimator(BaseEstimator):
    def __init__(self, agent, data_dict: dict, config: Config, args) -> None:
        super().__init__(agent, data_dict, config, args)
        done_indexs = np.where(self.dones == 1)[0]
        start_indexs = [0] + (done_indexs + 1).tolist()[:-1]
        self.initial_states = torch.tensor(self.states[start_indexs], dtype=torch.float)

    def estimate(self, **kwargs) -> Tuple[float, np.ndarray]:
        self.agent = kwargs['agent']
        states = self.initial_states.to(self.device)
        with torch.no_grad():
            if isinstance(self.agent, DQN):
                self.agent.model.eval()
                est_q_values, _ = self.agent.model(states).max(dim=1)
                est_q_values = est_q_values.view(1, -1).detach().cpu().numpy() # (B, 1)
            elif isinstance(self.agent, SAC):
                # weird because SAC's Q function contain entropy term
                actions = self.agent.get_action_probs(states)[0]
                qf1 = self.agent.qf1(states).gather(1, actions)
                qf2 = self.agent.qf2(states).gather(1, actions)
                est_q_values = torch.min(qf1, qf2).view(1, -1).detach().cpu().numpy()
        return est_q_values.mean(), est_q_values