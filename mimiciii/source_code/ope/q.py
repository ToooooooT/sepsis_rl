import numpy as np
import torch
from typing import Tuple

from utils import Config
from ope.base_estimator import BaseEstimator
from agents import DQN, SAC

class QEstimator(BaseEstimator):
    def __init__(self, agent, data_dict: dict, config: Config, args) -> None:
        super().__init__(agent, data_dict, config, args)

    def reset_data(self, data: dict[str, np.ndarray]):
        super().reset_data(data)
        self.initial_states = torch.tensor(self.states[self.start_indexs], dtype=torch.float)

    def estimate(self, **kwargs) -> Tuple[float, np.ndarray]:
        self.agent = kwargs['agent']
        states = self.initial_states.to(self.device)
        with torch.no_grad():
            if isinstance(self.agent, DQN):
                self.agent.q.eval()
                est_q_values, _ = self.agent.q(states).max(dim=1)
                est_q_values = est_q_values.detach().cpu().numpy() # (B,)
            elif isinstance(self.agent, SAC):
                self.agent.q_dre.eval()
                est_q_values, _ = self.agent.q_dre(states).max(dim=1)
                est_q_values = est_q_values.detach().cpu().numpy() # (B,)
        return est_q_values.mean(), est_q_values
