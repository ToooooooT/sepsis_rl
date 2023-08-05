import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
import joblib

def WIS_estimator(action_probs: np.ndarray, expert_data: pd.DataFrame, id_index_map, clip_expected_return):
    '''
    Args:
    action_probs    : policy action probabilities; expected shape (B, D)
    expert_data     : original expert dataset (DataFrame)
    id_index_map    : indexes of each icustayid (dict)
    Returns:
        avg_policy_return: average policy return
        policy_return: expected return of each patient; numpy array expected shape (1, B)
    '''
    # compute all trajectory total reward and weight imporatance sampling
    gamma = 0.99
    num = len(id_index_map)
    policy_return = np.zeros((num,), dtype=np.float64) 
    weights = np.zeros((num, 50)) # assume the patient max length is 50 
    length = np.zeros((num,), dtype=np.int32) # the horizon length of each patient
    rhos = action_probs[np.arange(action_probs.shape[0]), expert_data.loc[:, 'action'].values]
    for i, id in enumerate(id_index_map.keys()):
        start, end = id_index_map[id][0], id_index_map[id][-1]
        assert(50 >= end - start + 1)
        reward = 0
        length[i] = int(end - start + 1)
        for j, index in enumerate(range(end, start - 1, -1)):
            # let the minimum probability be 0.01 to avoid nan
            weights[i, end - start - j] = max(rhos[index], 0.01)
            # total reward
            reward = gamma * reward + expert_data.loc[index, 'reward']

        policy_return[i] = np.cumprod(weights[i])[length[i] - 1] * reward

    for i, l in enumerate(length):
        w_H = np.cumprod(weights[l <= length], axis=1)[:, l - 1].mean()
        policy_return[i] /= w_H

    policy_return = np.clip(policy_return, -clip_expected_return, clip_expected_return)
    return policy_return.mean(), policy_return.reshape(1, -1)


class sepsis_dataset(Dataset):
    def __init__(self, state: np.ndarray, action: np.ndarray, id_index_map, terminal_index) -> None:
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

        self.id_index_map = id_index_map
        self.terminal_index = terminal_index

    def __len__(self):
        return len(self.action) - 1

    def __getitem__(self, index):
        state = self.state[index]
        next_state = self.state[index + 1]
        state_action_pair = self.data[index]
        done = np.array([int(index in self.terminal_index)]) 
        return torch.tensor(state_action_pair, dtype=torch.float), \
            torch.tensor(next_state - state, dtype=torch.float), \
            torch.tensor(done, dtype=torch.float)


class DR_estimator():
    def __init__(self, test_dataset: pd.DataFrame, test_data_dict: dict, args, device) -> None:
        '''
        Args:
            train_data      : processed training dataset
            valid_data      : processed training dataset
            dataset         : unnormalization testing dataset, with action and reward 
            data_dict       : processed testing dataset
            args            : arguments from main file
        '''
        self.test_dataset = test_dataset
        self.test_data = test_data_dict['data']
        self.id_index_map = test_data_dict['id_index_map']
        self.terminal_index = test_data_dict['terminal_index']
        self.env_model = torch.load(args.env_model_path)['env']
        self.args = args
        self.device = device

        # train logistic regression to predict alive or not
        self.clf = joblib.load(args.clf_model_path)

        # for estimate reward
        sofa_mean = test_dataset['SOFA'].mean()
        sofa_std = test_dataset['SOFA'].std()
        lact_mean = test_dataset['Arterial_lactate'].mean()
        lact_std = test_dataset['Arterial_lactate'].std()
        norm_sofa = (test_dataset['SOFA'] - sofa_mean) / sofa_std
        norm_lact = (test_dataset['Arterial_lactate'] - lact_mean) / lact_std
        min_norm_sofa = norm_sofa.min()
        max_norm_sofa = norm_sofa.max()
        min_norm_lact = norm_lact.min()
        max_norm_lact = norm_lact.max()

        self.sofa_dicts = (sofa_std, sofa_mean, max_norm_sofa, min_norm_sofa)
        self.lact_dicts = (lact_std, lact_mean, max_norm_lact, min_norm_lact)


    def estimate_expected_return(self, 
                                 est_q_values: np.ndarray, 
                                 actions: np.ndarray, 
                                 action_probs: np.ndarray, 
                                 dataset: pd.DataFrame, 
                                 id_index_map: dict):
        '''
        Args:
            est_q_values    : estimate q value; expected shape: (B, 1) 
            actions         : policy action; expected shape: (B, 1)
            action_probs    : probability of policy action; expected shape: (B, 1)
            expert_data     : original expert dataset
            id_index_map    : indexes of each icustayid
        Returns:
            average policy return
            policy_return   : expected return of each patient; expected shape (1, B)
            est_alive       : estimate alive; expected shape (B,)
        '''
        # compute all trajectory total reward and weight imporatance sampling
        est_next_states = self.estimate_next_state(actions)
        est_reward, est_alive = self.estimate_reward(est_next_states, actions)
        gamma = 0.99
        num = len(id_index_map)
        policy_return = np.zeros((num,), dtype=np.float64) 
        expert_actions = dataset['action']
        for i, id in enumerate(id_index_map.keys()):
            start, end = id_index_map[id][0], id_index_map[id][-1]
            assert(50 >= end - start + 1)

            rho = action_probs[end, int(expert_actions[end])]
            reward = est_reward[end] + rho * (dataset.loc[end, 'reward'] - est_q_values[end])
            for index in range(end - 1, start - 1, -1):
                rho = action_probs[index, int(expert_actions[index])]
                reward = est_reward[index] + rho * (dataset.loc[end, 'reward'] + \
                                                    gamma * reward - est_q_values[index])
            policy_return[i] = reward

        policy_return = np.clip(policy_return, -self.args.clip_expected_return, self.args.clip_expected_return)
        return policy_return.mean(), policy_return.reshape(1, -1), est_alive


    def testing(self, input, label, done):
        input = input.to(self.device)
        label = label.to(self.device)
        done = done.to(self.device)

        with torch.no_grad():
            pred = self.env_model(input)
            loss = F.mse_loss(pred * (1 - done), label * (1 - done))
        return loss.detach().cpu().item(), pred.detach().cpu().numpy() + input[:, :-2].detach().cpu().numpy()


    def estimate_next_state(self, policy_actions: np.ndarray):
        '''
        Args:
            policy_actions: expected shape (B, 1)
        Returns:
            np.ndarray, expected shape (B, S) 
        '''
        test_loader = DataLoader(sepsis_dataset(self.test_data['s'], policy_actions, self.id_index_map, self.terminal_index),
                                batch_size=self.args.batch_size,
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True,
                                num_workers=self.args.num_worker)

        self.env_model.eval()
        test_loss = 0
        est_next_states = []
        for input, label, done in test_loader:
            loss, pred = self.testing(input, label, done)
            test_loss += loss
            est_next_states.append(pred)

        return np.vstack(est_next_states)


    def estimate_reward(self, est_next_states: np.ndarray, policy_actions: np.ndarray):
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
        state = self.test_data['s']
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
        icustayids = self.test_dataset['icustayid'].values
        lacts = self.test_dataset['Arterial_lactate'].values
        sofas = self.test_dataset['SOFA'].values
        for index in self.test_dataset.index:
            if index == len(self.test_dataset) - 1 or icustayids[index] != icustayids[index + 1]:
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
