######################################################################################
# Import Package
######################################################################################
import numpy as np
import pandas as pd
import torch
import os
import random
from argparse import ArgumentParser
from collections import defaultdict
from tqdm import tqdm

from agents import DQN, SAC, BaseAgent
from utils import Config, plot_action_dist, plot_action_distribution, plot_estimate_value, \
                animation_action_distribution, plot_pos_neg_action_dist, plot_diff_action_SOFA_dist, \
                plot_diff_action, plot_survival_rate
from network import DuellingMLP, PolicyMLP

from torch.utils.tensorboard import SummaryWriter

pd.options.mode.chained_assignment = None

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--hour", type=int, help="hours of one state", default=4)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=32)
    parser.add_argument("--episode", type=int, help="episode", default=70000)
    parser.add_argument("--use_pri", type=int, help="use priority replay", default=1)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-4)
    parser.add_argument("--reg_lambda", type=int, help="regularization term coeficient", default=5)
    parser.add_argument("--agent", type=str, help="agent type", default="D3QN")
    parser.add_argument("--test_dataset", type=str, help="test dataset", default="test")
    parser.add_argument("--valid_freq", type=int, help="validation frequency", default=50)
    parser.add_argument("--target_net_freq", type=int, help="the frequency of updates for the target networks", default=1)
    parser.add_argument("--seed", type=int, help="random seed", default=10)
    args = parser.parse_args()
    return args

######################################################################################
# Agent
######################################################################################
class D3QN_Agent(DQN):
    def __init__(self, static_policy=False, env=None, config=None, log_dir='./log', agent_dir='./saved_agents'):
        super().__init__(static_policy, env, config, log_dir, agent_dir)

    def declare_networks(self):
        self.model = DuellingMLP(self.num_feats, self.num_actions).to(self.device)
        self.target_model = DuellingMLP(self.num_feats, self.num_actions).to(self.device)


class SAC_Agent(SAC):
    def __init__(self, static_policy=False, env=None, config=None, log_dir='./log', agent_dir='./saved_agents'):
        super().__init__(static_policy, env, config, log_dir, agent_dir)

    def declare_networks(self):
        self.actor = PolicyMLP(self.num_feats, self.num_actions).to(self.device)
        self.qf1 = DuellingMLP(self.num_feats, self.num_actions).to(self.device)
        self.qf2 = DuellingMLP(self.num_feats, self.num_actions).to(self.device)
        self.target_qf1 = DuellingMLP(self.num_feats, self.num_actions).to(self.device)
        self.target_qf2 = DuellingMLP(self.num_feats, self.num_actions).to(self.device)


def add_dataset_to_replay(train, train_unnorm, model, clip_reward):
    # put all transitions in replay buffer
    drop_column = ['charttime', 'median_dose_vaso', 'input_total', 'icustayid', 'died_in_hosp', 'mortality_90d',
                'died_within_48h_of_out_time', 'delay_end_of_record_and_discharge_or_death',
                'input_4hourly', 'max_dose_vaso', 'reward', 'action']
    for i, index in tqdm(enumerate(train.index[:-1])):
        s = train.loc[index, :]
        s_ = train.loc[train.index[i + 1], :]
        a = s['action']
        r = s['reward']
        SOFA = train_unnorm.loc[index, 'SOFA']
        if s['icustayid'] != s_['icustayid']:
            s_ = None
        else:
            s_ = np.array(s_.drop(drop_column))
        s = np.array(s.drop(drop_column))
        if clip_reward and r != 15 and r != -15:
            if r > 1:
                r = 1
            if r < -1:
                r = -1
        model.append_to_replay(s, a, r, s_, SOFA)

    # handle last state
    idx = train.index[-1]
    s = train.loc[idx, :]
    a = s['action']
    r = s['reward']
    SOFA = train_unnorm.loc[idx, 'SOFA']
    s_ = None
    s = np.array(s.drop(drop_column))
    model.append_to_replay(s, a, r, s_, SOFA)


def process_dataset(dataset):
    drop_column = ['charttime', 'median_dose_vaso', 'input_total', 'icustayid', 'died_in_hosp', 'mortality_90d',
                'died_within_48h_of_out_time', 'delay_end_of_record_and_discharge_or_death',
                'input_4hourly', 'max_dose_vaso', 'reward', 'action']
    data = {'s': [], 'a': []}
    id_index_map = defaultdict(list)
    for index in tqdm(dataset.index):
        s = dataset.iloc[index, :]
        a = s['action']
        id_index_map[s['icustayid']].append(index)
        s.drop(drop_column, inplace=True)
        data['s'].append(s)
        data['a'].append(a)

    data['s'] = np.array(data['s'])
    data['a'] = np.array(data['a'])

    return data, id_index_map


def testing(test, model: BaseAgent):
    batch_state, batch_action = test['s'], test['a']

    batch_state = torch.tensor(batch_state, device=model.device, dtype=torch.float).view(-1, model.num_feats)
    batch_action = torch.tensor(batch_action, device=model.device, dtype=torch.int64).view(-1, 1)

    with torch.no_grad():
        if isinstance(model, D3QN_Agent):
            model.model.eval()
            actions = model.model(batch_state).max(dim=1)[1].view(-1, 1).cpu().numpy()
            ret = (actions, None)
        elif isinstance(model, SAC_Agent):
            model.actor.eval()
            actions, _, _, action_probs = model.actor.get_action(batch_state)
            actions = actions.view(-1, 1).cpu().numpy()
            ret = (actions, action_probs)
        
    model.save_action(actions)

    return ret


def training(model: BaseAgent, valid, config, valid_dataset, id_index_map, args):
    '''
    Args:
        valid           : batch_state and action (dict)
        valid_dataset   : original valid dataset (DataFrame)
        id_index_map    : indexes of each icustayid (dict)
    '''
    writer = SummaryWriter(model.log_dir)
    loss = 0
    avg_expert_returns = list()
    avg_policy_returns = list()
    hists = list() # save model actions of validation in every episode 
    valid_freq = args.valid_freq

    for i in tqdm(range(1, config.EPISODE + 1)):
        loss = model.update(i)
        writer.add_scalars('loss', loss, i)

        if i % valid_freq == 0:
            model.save()
            actions, action_probs = testing(valid, model)
            if i % 1000 == 0:
                hists.append(model.action_selections)
            avg_p_return, avg_e_return, _ = WIS_estimator(actions, action_probs, valid_dataset, id_index_map, args)
            avg_policy_returns.append(avg_p_return)
            avg_expert_returns.append(avg_e_return)

            writer.add_scalars('WIS_estimator', dict(zip(['learned', 'expert'], [avg_p_return, avg_e_return])), i)

    plot_action_distribution(model.action_selections, model.log_dir)
    animation_action_distribution(hists, model.log_dir)
    plot_estimate_value(avg_expert_returns, avg_policy_returns, model.log_dir)


def WIS_estimator(actions, action_probs, expert_data, id_index_map, args):
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
    policy_return = np.zeros((num,)) 
    expert_return = np.zeros((num,)) 
    weights = np.zeros((num,))
    for i, id in enumerate(id_index_map.keys()):
        start, end = id_index_map[id][0], id_index_map[id][-1]
        reward = 0
        ratio = 1
        for index in range(end, start - 1, -1):
            # assume policy take the max action in probability of 0.99 and any othe actions of 0.01 
            if args.agent == 'D3QN':
                ratio = ratio * 0.99 if int(actions[index]) == int(expert_data.loc[index, 'action']) else ratio * 0.01
            elif args.agent == 'SAC':
                # let the minimum probability be 0.01 to avoid nan
                ratio *= max(action_probs[index, int(expert_data.loc[index, 'action'])], 0.01)
            # total reward
            reward = gamma * reward + expert_data.loc[index, 'reward']

        weights[i] = ratio
        policy_return[i] = ratio * reward
        expert_return[i] = reward 

    avg_policy_return = (policy_return / weights.sum()).sum()
    avg_expert_return = expert_return.mean()
    return avg_policy_return, avg_expert_return, policy_return


if __name__ == '__main__':
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ######################################################################################
    # Load Dataset
    ######################################################################################
    dataset_path = "../data/final_dataset/"
    full_data = pd.read_csv(os.path.join(dataset_path, f'dataset_{args.hour}.csv'))
    train_dataset = pd.read_csv(os.path.join(dataset_path, f'train_{args.hour}.csv'))
    icustayids = train_dataset['icustayid'].unique()
    train_data_unnorm = full_data[[icustayid in icustayids for icustayid in full_data['icustayid']]]
    train_data_unnorm.index = range(train_dataset.shape[0])

    valid_dataset = pd.read_csv(os.path.join(dataset_path, f'valid_{args.hour}.csv'))

    test_dataset = pd.read_csv(os.path.join(dataset_path, f'{args.test_dataset}_{args.hour}.csv'))
    icustayids = test_dataset['icustayid'].unique()
    test_data_unnorm = full_data[[icustayid in icustayids for icustayid in full_data['icustayid']]]
    test_data_unnorm.index = range(test_dataset.shape[0])

    ######################################################################################
    # Hyperparameters
    ######################################################################################
    config = Config()

    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.EPISODE = args.episode
    config.USE_PRIORITY_REPLAY = args.use_pri
    config.LR = args.lr
    config.REG_LAMBDA = args.reg_lambda
    config.TARGET_NET_UPDATE_FREQ = args.target_net_freq
    config.BATCH_SIZE = args.batch_size

    # memory
    exp_replay_size = 1
    while exp_replay_size < train_dataset.shape[0]:
        exp_replay_size <<= 1

    config.EXP_REPLAY_SIZE = exp_replay_size

    clip_reward = True

    env = {'num_feats': 49, 'num_actions': 25}

    path = f'agent={args.agent}-batch_size={config.BATCH_SIZE}-episode={config.EPISODE}-use_pri={config.USE_PRIORITY_REPLAY}-lr={config.LR}-reg_lambda={config.REG_LAMBDA}-target_net_freq={config.TARGET_NET_UPDATE_FREQ}'
    log_path = os.path.join('./log', path)
    os.makedirs(log_path, exist_ok=True)
    agent_path = os.path.join('./saved_agents', path)
    os.makedirs(agent_path, exist_ok=True)

    if args.agent == 'D3QN':
        model = D3QN_Agent(static_policy=False, env=env, config=config, log_dir=log_path, agent_dir=agent_path)
    elif args.agent == 'SAC':
        model = SAC_Agent(static_policy=False, env=env, config=config, log_dir=log_path, agent_dir=agent_path)
    else:
        raise NotImplementedError

    ######################################################################################
    # Training
    ######################################################################################
    print('Adding dataset to replay buffer...')
    add_dataset_to_replay(train_dataset[:1000], train_data_unnorm[:1000], model, clip_reward)

    print('Processing validation dataset...')
    valid, id_index_map = process_dataset(valid_dataset)

    print('Start training...')
    training(model, valid, config, valid_dataset, id_index_map, args)

    ######################################################################################
    # Testing
    ######################################################################################
    model.load()

    print('Processing test dataset...')
    test, id_index_map = process_dataset(test_dataset)

    print('Start testing')
    actions, action_probs = testing(test, model)

    test_data_unnorm['action'] = test_dataset['action']
    test_data_unnorm['policy action'] = actions
    test_data_unnorm.to_csv(os.path.join(model.log_dir, 'test_data_predict.csv'), index=False)
    negative_traj = test_data_unnorm.query('died_in_hosp == 1.0 | died_within_48h_of_out_time == 1.0 | mortality_90d == 1.0')
    positive_traj = test_data_unnorm.query('died_in_hosp != 1.0 & died_within_48h_of_out_time != 1.0 & mortality_90d != 1.0')
    avg_policy_return, avg_expert_return, policy_return = WIS_estimator(actions, action_probs, test_dataset, id_index_map, args)
    plot_action_dist(actions, test_data_unnorm, model.log_dir)
    plot_pos_neg_action_dist(positive_traj, negative_traj, log_path)
    plot_diff_action_SOFA_dist(positive_traj, negative_traj, log_path)
    plot_diff_action(positive_traj, negative_traj, log_path)
    plot_survival_rate(policy_return, id_index_map, test_data_unnorm, log_path)
    with open(os.path.join(log_path, 'evaluation.txt'), 'w') as f:
        f.write(f'policy WIS estimator: {avg_policy_return:.5f}\n')
        f.write(f'expert: {avg_expert_return:.5f}')
    print(f'policy WIS estimator: {avg_policy_return:.5f}')
    print(f'expert: {avg_expert_return:.5f}')