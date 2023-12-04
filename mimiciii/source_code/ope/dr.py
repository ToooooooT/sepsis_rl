import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
import joblib

from ope.base_estimator import BaseEstimator
from utils import Config
from agents import BaseAgent, DQN, SAC

class SepsisDataset(Dataset):
    def __init__(self, state: np.ndarray, action: np.ndarray) -> None:
        '''
        Args:
            state: expected shape (B, S)
            action: expected shape (B, 1)
        '''
        super().__init__()
        self.state_dim = state.shape[1]
        # add a zero in last row to avoid overflow of index in next state
        state = np.vstack((state, np.zeros((1, self.state_dim))))
        self.state = state

        self.action_dim = 2
        # add a zero in last row to avoid overflow of index in next state
        self.action = np.array([(x // 5, x % 5) for x in action]).reshape(-1, 2)
        self.action = np.vstack((self.action, np.zeros((1, self.action_dim))))

        self.data = np.concatenate([self.state, self.action], axis=1)

    def __len__(self):
        return len(self.action) - 1

    def __getitem__(self, index):
        state_action_pair = self.data[index]
        return torch.tensor(state_action_pair, dtype=torch.float)


class DoublyRobust(BaseEstimator):
    def __init__(self, 
                 agent: BaseAgent,
                 data_dict: dict, 
                 config: Config,
                 args,
                 dataset: pd.DataFrame) -> None:
        super().__init__(agent, data_dict, config, args)
        self.dataset = dataset
        self.batch_size = config.BATCH_SIZE
        self.num_worker = args.num_worker

        # train an environment to predict next state
        self.env_model = torch.load(args.env_model_path)['env']

        # train logistic regression to predict alive or not
        self.clf = joblib.load(args.clf_model_path)

        # for estimate reward
        sofa_mean = dataset['SOFA'].mean()
        sofa_std = dataset['SOFA'].std()
        lact_mean = dataset['Arterial_lactate'].mean()
        lact_std = dataset['Arterial_lactate'].std()
        norm_sofa = (dataset['SOFA'] - sofa_mean) / sofa_std
        norm_lact = (dataset['Arterial_lactate'] - lact_mean) / lact_std
        min_norm_sofa = norm_sofa.min()
        max_norm_sofa = norm_sofa.max()
        min_norm_lact = norm_lact.min()
        max_norm_lact = norm_lact.max()

        self.sofa_dicts = (sofa_std, sofa_mean, max_norm_sofa, min_norm_sofa)
        self.lact_dicts = (lact_std, lact_mean, max_norm_lact, min_norm_lact)

    def estimate_q_values(self):
        states = torch.tensor(self.states, dtype=torch.float, device=self.device)
        with torch.no_grad():
            if isinstance(self.agent, DQN):
                self.agent.model.eval()
                est_q_values, _ = self.agent.model(states).max(dim=1)
                est_q_values = est_q_values.view(-1, 1).detach().cpu().numpy() # (B, 1)
            elif isinstance(self.agent, SAC):
                # weird because SAC's Q function contain entropy term
                actions = self.agent.get_action_probs(states)[0]
                qf1 = self.agent.qf1(states).gather(1, actions)
                qf2 = self.agent.qf2(states).gather(1, actions)
                est_q_values = torch.min(qf1, qf2).view(-1, 1).detach().cpu().numpy()
        return est_q_values


    def estimate(self, **kwargs):
        '''
        Args:
            policy_actions     : np.ndarray; expected shape (B, 1)
            policy_action_probs: np.ndarray; expected shape (B, D)
        Returns:
            average policy return
            policy_return   : expected return of each patient; expected shape (1, B)
            est_alive       : estimate alive; expected shape (B,)
        '''
        self.agent = kwargs['agent']
        policy_actions = kwargs['policy_actions']
        policy_action_probs = kwargs['policy_action_probs']
        est_q_values = self.estimate_q_values()

        est_next_states = self.estimate_next_state(policy_actions)
        est_reward, est_alive = self.estimate_reward(est_next_states, policy_actions)

        # \rho_t = \pi_1(a_t | s_t) / \pi_0(a_t | s_t), assume \pi_0(a_t | s_t) = 1
        rhos = policy_action_probs[np.arange(policy_action_probs.shape[0]), 
                                   self.actions.astype(np.int32).reshape(-1,)]

        # done index
        done_indexs = np.where(self.dones == 1)[0]

        num = done_indexs.shape[0]
        policy_return = np.zeros((num,), dtype=np.float64) 

        start = 0
        for i in range(done_indexs.shape[0]):
            end = done_indexs[i]
            total_reward = est_reward[end] + rhos[end] * (self.rewards[end] - est_q_values[end])
            for index in range(end - 1, start - 1, -1):
                total_reward = est_reward[index] + rhos[index] * (self.rewards[index] + \
                                                    self.gamma * total_reward - est_q_values[index])
            start = end + 1
            policy_return[i] = total_reward

        policy_return = np.clip(policy_return, -self.clip_expected_return, self.clip_expected_return)
        return policy_return.mean(), policy_return.reshape(1, -1), est_alive


    def estimate_next_state(self, policy_actions: np.ndarray):
        '''
        Args:
            policy_actions: expected shape (B, 1)
        Returns:
            np.ndarray, expected shape (B, S) 
        '''
        sepsis_dataset = SepsisDataset(self.states, policy_actions)
        test_loader = DataLoader(sepsis_dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True,
                                num_workers=self.num_worker)
        self.env_model.eval()
        est_next_states = []
        for input in test_loader:
            with torch.no_grad():
                pred = self.env_model(input)
            est_next_state = pred.detach().cpu().numpy() + input[:, :-2].detach().cpu().numpy()
            est_next_states.append(est_next_state)

        return np.vstack(est_next_states)


    def estimate_reward(self, 
                        est_next_states: np.ndarray, 
                        policy_actions: np.ndarray):
        '''
        Args:
            policy_actions: expected shape: (B, 1)
            est_next_states: next state estimate from environment model;
                                SOFA: -2 (index), lactate: -10 (index)
        Returns:
            est_rewards: np.ndarray, expected shape: (B,)
            est_alive: np.ndarray, expected shape: (B,)
        '''
        iv = policy_actions / 5
        vaso = policy_actions % 5
        state = self.states
        clf_feat = np.concatenate([state, iv, vaso], axis=1)
        c0 = -0.1 / 4
        c1 = -0.5 / 4
        c2 = -2

        sofa_std, sofa_mean, max_norm_sofa, min_norm_sofa = self.sofa_dicts
        lact_std, lact_mean, max_norm_lact, min_norm_lact = self.lact_dicts

        est_next_sofa = est_next_states[:, -2]
        est_next_lact = est_next_states[:, -10]
        est_next_sofa = sofa_std * (est_next_sofa * (max_norm_sofa - min_norm_sofa) + min_norm_sofa) + sofa_mean
        est_next_lact = lact_std * (est_next_lact * (max_norm_lact - min_norm_lact) + min_norm_lact) + lact_mean

        est_rewards = []
        est_alives = []
        icustayids = self.dataset['icustayid'].values
        lacts = self.dataset['Arterial_lactate'].values
        sofas = self.dataset['SOFA'].values
        for index in self.dataset.index:
            if index == len(self.dataset) - 1 or icustayids[index] != icustayids[index + 1]:
                terminal_feat = clf_feat[index, :].reshape(1, -1)
                est_outcome = self.clf.predict(terminal_feat)[0]
                est_alives.append(est_outcome)
                est_rewards.append(15 if est_outcome == 1 else -15)
            else:
                lact_cur = lacts[index]
                sofa_cur = sofas[index]
                lact_next = est_next_lact[index]
                sofa_next = est_next_sofa[index]
                reward = 0
                if sofa_next == sofa_cur and sofa_next != 0:
                    reward += c0
                reward += c1 * (sofa_next - sofa_cur)
                reward += c2 * np.tanh(lact_next - lact_cur)
                est_rewards.append(reward)

        return np.array(est_rewards), np.array(est_alives)
