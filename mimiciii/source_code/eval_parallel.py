######################################################################################
# Import Package
######################################################################################
import numpy as np
import pandas as pd
import torch
import os
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from collections import defaultdict
from multiprocessing import Process

from agents.DQN import Model as DQN_Agent
from utils.hyperparameters import Config
from network.D3QN import D3QN

pd.options.mode.chained_assignment = None

######################################################################################
# Agent
######################################################################################
class Model(DQN_Agent):
    def __init__(self, static_policy=False, env=None, config=None, log_dir='./log'):
        super().__init__(static_policy, env, config, log_dir)

    def declare_networks(self):
        self.model = D3QN(self.num_feats, self.num_actions)
        self.target_model = D3QN(self.num_feats, self.num_actions)


def get_test_dataset(test_dataset):
    drop_column = ['charttime', 'median_dose_vaso', 'input_total', 'icustayid', 'died_in_hosp', 'mortality_90d',
                'died_within_48h_of_out_time', 'delay_end_of_record_and_discharge_or_death',
                'input_4hourly', 'max_dose_vaso', 'reward', 'action']
    test = {'s': [], 'a': []}
    id_index_map = defaultdict(list)
    for index in test_dataset.index:
        s = test_dataset.iloc[index, :]
        a = s['action']
        id_index_map[s['icustayid']].append(index)
        s.drop(drop_column, inplace=True)
        test['s'].append(s)
        test['a'].append(a)

    test['s'] = np.array(test['s'])
    test['a'] = np.array(test['a'])

    return test, id_index_map
    

def testing(test, model):
    batch_state, batch_action = test['s'], test['a']

    batch_state = torch.tensor(batch_state, device=model.device, dtype=torch.float).view(-1, model.num_feats)
    batch_action = torch.tensor(batch_action, device=model.device, dtype=torch.int64).view(-1, 1)

    with torch.no_grad():
        actions = model.model(batch_state).max(dim=1)[1].view(-1, 1)
        
    model.save_action(actions)

    return actions


def plot_action_dist(model, actions, test_data_unnorm):
    '''
    actions                 : policy action (tensor)
    test_data_unnorm        : original expert dataset unnormalize (DataFrame)
    '''
    actions_low = [0] * 25
    actions_mid = [0] * 25
    actions_high = [0] * 25
    actions_all = [0] * 25
    for index in test_data_unnorm.index:
        # action distribtuion
        if test_data_unnorm.loc[index, 'SOFA'] <= 5:
            actions_low[int(actions[index]) - 1] += 1
        elif test_data_unnorm.loc[index, 'SOFA'] < 15:
            actions_mid[int(actions[index]) - 1] += 1
        else:
            actions_high[int(actions[index]) - 1] += 1
        actions_all[int(actions[index]) - 1] += 1

    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16,4))
    ax1.hist(range(25), weights=actions_low, bins=25)
    ax1.set_xticks(range(0, 25))
    ax1.tick_params(axis='x', labelsize=6)
    ax1.set_title('low SOFA')

    ax2.hist(range(25), weights=actions_mid, bins=25)
    ax2.set_xticks(range(0, 25))
    ax2.tick_params(axis='x', labelsize=6)
    ax2.set_title('mid SOFA')

    ax3.hist(range(25), weights=actions_high, bins=25)
    ax3.set_xticks(range(0, 25))
    ax3.tick_params(axis='x', labelsize=6)
    ax3.set_title('high SOFA')

    ax4.hist(range(25), weights=actions_all, bins=25)
    ax4.set_xticks(range(0, 25))
    ax4.tick_params(axis='x', labelsize=6)
    ax4.set_title('all')

    plt.savefig(os.path.join(model.log_dir, 'D3QN test action distribution.png'))


def WIS_estimator(actions, test_data, id_index_map):
    '''
    actions                 : policy action (tensor)
    test_data               : original expert dataset (DataFrame)
    id_index_map            : indexes of each icustayid (dict)
    return ->
        policy_val          : estimate policy value
        expert_val          : estimate expert value
    '''
    # compute all trajectory total reward and weight imporatance sampling
    gamma = 0.99
    weight = 0
    policy_val = expert_val = 0
    for id in id_index_map.keys():
        start, end = id_index_map[id][0], id_index_map[id][-1]
        reward = 0
        ratio = 1
        for index in range(end, start - 1, -1):
            # assume policy take the max action in probability of 0.99 and any othe actions of 0.01 
            ratio = ratio * 0.99 if int(actions[index]) == int(test_data.loc[index, 'action']) else ratio * 0.01
            # total reward
            reward = gamma * reward + test_data.loc[index, 'reward']
            
        policy_val += ratio * reward
        expert_val += reward
        weight += ratio

    n = len(id_index_map.keys())
    weight /= n
    policy_val = policy_val / n / weight
    expert_val /= n
    return policy_val, expert_val


def test_parallel(config, env, lr, test_data, test, id_index_map):
    config.LR = lr / 10000

    path = os.path.join('./log', f'batch_size-{config.BATCH_SIZE} episode-{config.EPISODE} use_pri-{config.USE_PRIORITY_REPLAY} lr-{config.LR} reg_lambda-{config.REG_LAMBDA}')
    if not os.path.exists(path):
        os.mkdir(path)

    model = Model(static_policy=True, env=env, config=config, log_dir=path)

    model.load()

    ######################################################################################
    # Testing
    ######################################################################################
    actions = testing(test, model)

    policy_val, expert_val = WIS_estimator(actions, test_data, id_index_map)
    plot_action_dist(model, actions, test_data_unnorm)
    with open(os.path.join(path, 'evaluation.txt'), 'w') as f:
        f.write(f'policy WIS estimator: {policy_val:.5f}\n')
        f.write(f'expert: {expert_val:.5f}')
    print(f'policy WIS estimator: {policy_val:.5f}')
    print(f'expert: {expert_val:.5f}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--hour", type=int, help="hours of one state", dest="hour", default=4)
    parser.add_argument("--batch_size", type=int, help="batch_size", dest="batch_size", default=32)
    parser.add_argument("--episode", type=int, help="episode", dest="episode", default=70000)
    parser.add_argument("--use_pri", type=int, help="use priority replay", dest="use_pri", default=1)
    parser.add_argument("--lr", type=float, help="learning rate", dest="lr", default=0.0001)
    parser.add_argument("--reg_lambda", type=float, help="regularization term coeficient", dest="reg_lambda", default=5)
    args = parser.parse_args()
    hour = args.hour
    batch_size = args.batch_size
    episode = args.episode
    use_pri = args.use_pri
    lr = args.lr
    reg_lambda = args.reg_lambda

    ######################################################################################
    # Load Dataset
    ######################################################################################
    dataset_path = "../data/final_dataset/"

    test_data = pd.read_csv(os.path.join(dataset_path, f'test_{hour}.csv'))
    test_data_unnorm = pd.read_csv(os.path.join(dataset_path, f'dataset_{hour}.csv'))
    icustayids = test_data['icustayid'].unique()
    test_data_unnorm = test_data_unnorm[[icustayid in icustayids for icustayid in test_data_unnorm['icustayid']]]
    test_data_unnorm.index = range(test_data.shape[0])

    ######################################################################################
    # Parameters
    ######################################################################################
    config = Config()

    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.BATCH_SIZE = batch_size
    config.USE_PRIORITY_REPLAY = use_pri
    config.EPISODE = episode
    config.REG_LAMBDA = reg_lambda
    # config.LR = lr

    env = {'num_feats': 49, 'num_actions': 25}

    ######################################################################################
    # Parallel test
    ######################################################################################
    test, id_index_map = get_test_dataset(test_data)

    processes = list()
    for lr in range(1, 11):
        processes.append(Process(target=test_parallel, args=(config, env, lr, test_data, test, id_index_map)))
    for i in range(len(processes)):
        processes[i].start()
    for i in range(len(processes)):
        processes[i].join()