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

from agents import Model as DQN_Agent
from utils import Config, plot_action_dist, plot_action_distribution, plot_estimate_value, animation_action_distribution
from network import DuellingMLP

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
    parser.add_argument("--valid_freq", type=int, help="validation frequency", default=10)
    parser.add_argument("--target_net_freq", type=int, help="the frequency of updates for the target networks", default=1)
    parser.add_argument("--seed", type=int, help="random seed", default=10)
    args = parser.parse_args()
    return args

######################################################################################
# Agent
######################################################################################
class Model(DQN_Agent):
    def __init__(self, static_policy=False, env=None, config=None, log_dir='./log', agent_dir='./saved_agents'):
        super().__init__(static_policy, env, config, log_dir, agent_dir)

    def declare_networks(self):
        self.model = DuellingMLP(self.num_feats, self.num_actions).to(self.device)
        self.target_model = DuellingMLP(self.num_feats, self.num_actions).to(self.device)


def add_dataset_to_replay(train, model, clip_reward):
    # put all transitions in replay buffer
    drop_column = ['charttime', 'median_dose_vaso', 'input_total', 'icustayid', 'died_in_hosp', 'mortality_90d',
                'died_within_48h_of_out_time', 'delay_end_of_record_and_discharge_or_death',
                'input_4hourly', 'max_dose_vaso', 'reward', 'action']
    for i, index in tqdm(enumerate(train.index[:-1])):
        s = train.loc[index, :]
        s_ = train.loc[train.index[i + 1], :]
        a = s['action']
        r = s['reward']
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
        model.append_to_replay(s, a, r, s_)

    # handle last state
    idx = train.index[-1]
    s = train.loc[idx, :]
    a = s['action']
    r = s['reward']
    s_ = None
    s = np.array(s.drop(drop_column))
    model.append_to_replay(s, a, r, s_)


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


def testing(test, model: Model):
    batch_state, batch_action = test['s'], test['a']

    batch_state = torch.tensor(batch_state, device=model.device, dtype=torch.float).view(-1, model.num_feats)
    batch_action = torch.tensor(batch_action, device=model.device, dtype=torch.int64).view(-1, 1)

    with torch.no_grad():
        actions = model.model(batch_state).max(dim=1)[1].view(-1, 1).cpu().numpy()
        
    model.save_action(actions)

    return actions


def training(model: Model, valid, config, valid_dataset, id_index_map, args):
    '''
    Args:
        valid           : batch_state and action (dict)
        valid_dataset   : original valid dataset (DataFrame)
        id_index_map    : indexes of each icustayid (dict)
    Returns:
        policy_val  : list of policy estimate value
        expert_val  : list of expert value
    '''
    writer = SummaryWriter(model.log_dir)
    loss = 0
    expert_val = list()
    policy_val = list()
    hists = list() # save model actions of validation in every episode 
    valid_freq = args.valid_freq

    for i in tqdm(range(1, config.EPISODE + 1)):
        loss = model.update(i)
        writer.add_scalars('loss', loss, i)

        if i % valid_freq == 0:
            model.save()
            actions = testing(valid, model)
            if i % 1000 == 0:
                hists.append(model.action_selections)
            p_val, e_val = WIS_estimator(actions, valid_dataset, id_index_map)
            policy_val.append(p_val)
            expert_val.append(e_val)

            writer.add_scalars('WIS_estimator', dict(zip(['learned', 'expert'], [p_val, e_val])), i)

    plot_action_distribution(model.action_selections, model.log_dir)
    animation_action_distribution(hists, model.log_dir)
    plot_estimate_value(expert_val, policy_val, model.log_dir)


def WIS_estimator(actions, expert_data, id_index_map):
    '''
    Args:
    actions         : policy action (tensor)
    expert_data     : original expert dataset (DataFrame)
    id_index_map    : indexes of each icustayid (dict)
    Returns:
        policy_val  : policy estimation value
        expert_val  : expert value
    '''
    # compute all trajectory total reward and weight imporatance sampling
    gamma = 0.99
    policy_val = expert_val = 0
    weight = 0
    for id in id_index_map.keys():
        start, end = id_index_map[id][0], id_index_map[id][-1]
        reward = 0
        ratio = 1
        for index in range(end, start - 1, -1):
            # assume policy take the max action in probability of 0.99 and any othe actions of 0.01 
            ratio = ratio * 0.99 if int(actions[index]) == int(expert_data.loc[index, 'action']) else ratio * 0.01
            # total reward
            reward = gamma * reward + expert_data.loc[index, 'reward']

        weight += ratio
        policy_val += ratio * reward
        expert_val += reward 

    n = len(id_index_map.keys())
    weight /= n
    policy_val = policy_val / n / weight
    expert_val /= n
    return policy_val, expert_val


if __name__ == '__main__':
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ######################################################################################
    # Load Dataset
    ######################################################################################
    dataset_path = "../data/final_dataset/"
    train_dataset = pd.read_csv(os.path.join(dataset_path, f'train_{args.hour}.csv'))
    valid_dataset = pd.read_csv(os.path.join(dataset_path, f'valid_{args.hour}.csv'))

    test_data = pd.read_csv(os.path.join(dataset_path, f'{args.test_dataset}_{args.hour}.csv'))
    test_data_unnorm = pd.read_csv(os.path.join(dataset_path, f'dataset_{args.hour}.csv'))
    icustayids = test_data['icustayid'].unique()
    test_data_unnorm = test_data_unnorm[[icustayid in icustayids for icustayid in test_data_unnorm['icustayid']]]
    test_data_unnorm.index = range(test_data.shape[0])

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

    path = f'agent={args.agent}-batch_size={config.BATCH_SIZE}-episode={config.EPISODE}-use_pri={config.USE_PRIORITY_REPLAY}-lr={config.LR}-reg_lambda={config.REG_LAMBDA}-target_net_freq{config.TARGET_NET_UPDATE_FREQ}'
    log_path = os.path.join('./log', path)
    os.makedirs(log_path, exist_ok=True)
    agent_path = os.path.join('./saved_agents', path)
    os.makedirs(agent_path, exist_ok=True)

    model = Model(static_policy=False, env=env, config=config, log_dir=log_path, agent_dir=agent_path)

    ######################################################################################
    # Training
    ######################################################################################
    print('Adding dataset to replay buffer...')
    add_dataset_to_replay(train_dataset, model, clip_reward)

    print('Processing validation dataset...')
    valid, id_index_map = process_dataset(valid_dataset)

    print('Start training...')
    training(model, valid, config, valid_dataset, id_index_map, args)

    ######################################################################################
    # Testing
    ######################################################################################
    model.load()

    print('Processing test dataset...')
    test, id_index_map = process_dataset(test_data)

    print('Start testing')
    actions = testing(test, model)

    test_data_unnorm['action'] = test_data['action']
    test_data_unnorm['policy action'] = actions
    test_data_unnorm.to_csv(os.path.join(model.log_dir, 'test_data_predict.csv'), index=False)
    policy_val, expert_val = WIS_estimator(actions, test_data, id_index_map)
    plot_action_dist(actions, test_data_unnorm, model.log_dir)
    with open(os.path.join(log_path, 'evaluation.txt'), 'w') as f:
        f.write(f'policy WIS estimator: {policy_val:.5f}\n')
        f.write(f'expert: {expert_val:.5f}')
    print(f'policy WIS estimator: {policy_val:.5f}')
    print(f'expert: {expert_val:.5f}')