import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

def WIS_estimator(actions, action_probs, expert_data: pd.DataFrame, id_index_map, args):
    '''
    Args:
    actions         : policy action (tensor)
    expert_data     : original expert dataset (DataFrame)
    id_index_map    : indexes of each icustayid (dict)
    Returns:
        avg_policy_return: average policy return
        avg_expert_return: average expert return
        policy_return: expected return of each patient; expected shape (B,)
    '''
    # compute all trajectory total reward and weight imporatance sampling
    gamma = 0.99
    num = len(id_index_map)
    policy_return = np.zeros((num,), dtype=np.float64) 
    weights = np.zeros((num, 50)) # assume the patient max length is 50 
    length = np.zeros((num,), dtype=np.int32) # the horizon length of each patient
    for i, id in enumerate(id_index_map.keys()):
        start, end = id_index_map[id][0], id_index_map[id][-1]
        assert(50 >= end - start + 1)
        reward = 0
        length[i] = int(end - start + 1)
        for j, index in enumerate(range(end, start - 1, -1)):
            # assume policy take the max action in probability of 0.99 and any other actions of 0.01 
            if args.agent == 'D3QN':
                weights[i, end - start - j] = 0.99 if int(actions[index]) == int(expert_data.loc[index, 'action']) else 0.01
            elif args.agent == 'SAC':
                # let the minimum probability be 0.01 to avoid nan
                weights[i, end - start - j] = max(action_probs[index, int(expert_data.loc[index, 'action'])], 0.01)
            # total reward
            reward = gamma * reward + expert_data.loc[index, 'reward']

        policy_return[i] = np.cumprod(weights[i])[length[i] - 1] * reward

    for i, l in enumerate(length):
        w_H = np.cumprod(weights[l <= length], axis=1)[:, l - 1].mean()
        policy_return[i] /= w_H

    policy_return = np.clip(policy_return, -40, 40)
    return policy_return.mean(), policy_return


class sepsis_dataset(Dataset):
    def __init__(self, state, action, id_index_map, terminal_index) -> None:
        super().__init__()
        self.state_dim = state.shape[1]
        # add a zero in last row to avoid overflow of index in next state
        state = np.vstack((state, np.zeros((1, self.state_dim))))
        self.state = state

        self.action_dim = 2
        # add a zero in last row to avoid overflow of index in next state
        self.action = np.array([(x // 5, x % 5) for x in action])
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
    def __init__(self, train_feat, train_labels, unnorm_dataset: pd.DataFrame, data_dict, args) -> None:
        '''
        Args:
            train_feat      : training dataset states
            train_labels    : training dataset alive or not
            dataset         : unnormalization dataset, with action and reward 
            data_dict       : pickle file
            args            : arguments from main file
        '''
        self.unnorm_dataset = unnorm_dataset
        self.data = data_dict['data']
        self.id_index_map = data_dict['id_index_map']
        self.terminal_index = data_dict['terminal_index']
        self.env_model = torch.load(args.env_model_path)['env']

        self.args = args

        self.clf = LogisticRegression()
        self.clf.fit(train_feat, train_labels)

        # for estimate reward
        sofa_mean = unnorm_dataset['SOFA'].mean()
        sofa_std = unnorm_dataset['SOFA'].std()
        lact_mean = unnorm_dataset['Arterial_lactate'].mean()
        lact_std = unnorm_dataset['Arterial_lactate'].std()

        norm_sofa = (unnorm_dataset['SOFA'] - sofa_mean) / sofa_std
        norm_lact = (unnorm_dataset['Arterial_lactate'] - lact_mean) / lact_std

        min_norm_sofa = norm_sofa.min()
        max_norm_sofa = norm_sofa.max()

        min_norm_lact = norm_lact.min()
        max_norm_lact = norm_lact.max()

        self.sofa_dicts = (sofa_std, sofa_mean, max_norm_sofa, min_norm_sofa)
        self.lact_dicts = (lact_std, lact_mean, max_norm_lact, min_norm_lact)


    def estimate_expected_return(self, est_reward, est_q_value, action_probs, expert_data, id_index_map):
        '''
        Args:
        est_next_state  : estimate next state
        est_q_value     : estimate q value
        actions         : policy action (tensor)
        expert_data     : original expert dataset (DataFrame)
        id_index_map    : indexes of each icustayid (dict)
        Returns:
            average policy return
            policy_return: expected return of each patient; expected shape (B,)
        '''
        # compute all trajectory total reward and weight imporatance sampling
        gamma = 0.99
        num = len(id_index_map)
        policy_return = np.zeros((num,), dtype=np.float64) 
        expert_actions = expert_data['action']
        for i, id in enumerate(id_index_map.keys()):
            start, end = id_index_map[id][0], id_index_map[id][-1]
            assert(50 >= end - start + 1)

            rho = action_probs[end, int(expert_actions[end])]
            reward = est_reward[end] + rho * (expert_data.loc[end, 'reward'] - est_q_value[end])
            for index in range(end - 1, start - 1, -1):
                rho = action_probs[index, int(expert_actions[index])]
                reward = est_reward[index] + rho * (expert_data.loc[end, 'reward'] + \
                                                    gamma * reward - est_q_value[index])
            policy_return[i] = reward

        return policy_return[i].mean(), policy_return


    def testing(self, input, label, done, device):
        input = input.to(device)
        label = label.to(device)
        done = done.to(device)

        with torch.no_grad():
            pred = self.env_model(input)
            loss = F.mse_loss(pred * (1 - done), label * (1 - done))
        return loss.detach().cpu().item(), pred.detach().cpu().numpy() + input[:, :-2].detach().cpu().numpy()


    def estimate_next_state(self, policy_actions):
        test_loader = DataLoader(sepsis_dataset(self.data['s'], policy_actions, self.id_index_map, self.terminal_index),
                                batch_size=self.args.batch_size,
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True,
                                num_workers=self.args.num_worker)

        self.env_model.eval()
        test_loss = 0
        est_next_states = []
        for input, label, done in test_loader:
            loss, pred = self.testing(input, label, done, self.args.device)
            test_loss += loss
            est_next_states.append(pred)

        return np.array(est_next_states)


    def estimate_reward(self, est_next_states):
        '''
        Args:
            est_next_states: next state estimate from environment model;
                                SOFA: -2 (index), lactate: -10 (index)
        '''
        c0 = -0.1 / 4
        c1 = -0.5 / 4
        c2 = -2

        sofa_std, sofa_mean, max_norm_sofa, min_norm_sofa = self.sofa_dicts
        lact_std, lact_mean, max_norm_lact, min_norm_lact = self.lact_dicts

        est_next_sofa = est_next_states[:, -2]
        est_next_lact = est_next_states[:, -10]
        est_next_sofa = sofa_std * (est_next_sofa * (max_norm_sofa - min_norm_sofa) + min_norm_sofa) + sofa_mean
        est_next_lact = lact_std * (est_next_lact * (max_norm_lact - min_norm_lact) + min_norm_lact) + lact_mean

        rewards = []
        for index in self.unnorm_dataset.index:
            if index == len(self.unnorm_dataset) - 1 or self.unnorm_dataset.loc[index, 'icustay_id'] \
                != self.unnorm_dataset.loc[index + 1, 'icustay_id']:
                terminal_state = self.data['s'][index, :]
                est_outcome = self.clf.predict(terminal_state)
                try:
                    rewards.append(15 if est_outcome == 1 else -15)
                except:
                    print(f'estimate outcome: {est_outcome}')
                    raise ValueError
            else:
                lact_cur = self.unnorm_dataset.loc[index, 'Arterial_lactate']
                sofa_cur = self.unnorm_dataset.loc[index, 'SOFA']
                lact_next = est_next_lact[index]
                sofa_next = est_next_sofa[index]
                reward = 0
                if sofa_next == sofa_cur and sofa_next != 0:
                    reward += c0
                reward += c1 * (sofa_next - sofa_cur)
                reward += c2 * np.tanh(lact_next - lact_cur)
                rewards.append(reward)

        return np.array(rewards)
