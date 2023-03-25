######################################################################################
# Import Package
######################################################################################
import numpy as np
import pandas as pd
import torch
import os
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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
    def __init__(self, static_policy=False, env=None, config=None, log_dir='./log', agent_dir='saved_agents'):
        super().__init__(static_policy, env, config, log_dir, agent_dir)

    def declare_networks(self):
        self.model = D3QN(self.num_feats, self.num_actions)
        self.target_model = D3QN(self.num_feats, self.num_actions)

######################################################################################
# Training Loop
######################################################################################
def add_dataset_to_replay(train, model):
    # put all transitions in replay buffer
    drop_column = ['charttime', 'median_dose_vaso', 'input_total', 'icustayid', 'died_in_hosp', 'mortality_90d',
                'died_within_48h_of_out_time', 'delay_end_of_record_and_discharge_or_death',
                'input_4hourly', 'max_dose_vaso', 'reward', 'action']
    for i, index in enumerate(train.index[:-1]):
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


def get_valid_dataset(valid_dataset):
    drop_column = ['charttime', 'median_dose_vaso', 'input_total', 'icustayid', 'died_in_hosp', 'mortality_90d',
                'died_within_48h_of_out_time', 'delay_end_of_record_and_discharge_or_death',
                'input_4hourly', 'max_dose_vaso', 'reward', 'action']
    validation = {'s': [], 'a': []}
    id_index_map = defaultdict(list)
    for index in valid_dataset.index:
        s = valid_dataset.iloc[index, :]
        a = s['action']
        id_index_map[s['icustayid']].append(index)
        s.drop(drop_column, inplace=True)
        validation['s'].append(s)
        validation['a'].append(a)

    validation['s'] = np.array(validation['s'])
    validation['a'] = np.array(validation['a'])

    return validation, id_index_map


def validation(valid_dataset, model):
    batch_state, batch_action = valid_dataset['s'], valid_dataset['a']

    batch_state = torch.tensor(batch_state, device=model.device, dtype=torch.float).view(-1, model.num_feats)
    batch_action = torch.tensor(batch_action, device=model.device, dtype=torch.int64).view(-1, 1)

    with torch.no_grad():
        actions = model.model(batch_state).max(dim=1)[1].view(-1, 1)
        
    model.save_action(actions)

    return actions


def training(model, valid, config, valid_dataset, id_index_map):
    '''
    valid           : batch_state and action (dict)
    valid_dataset   : original valid dataset (DataFrame)
    id_index_map    : indexes of each icustayid (dict)
    return ->
        policy_val  : list of policy estimate value
        expert_val  : list of expert value
    '''

    loss = 0
    expert_val = list()
    policy_val = list()
    hists = list() # save model actions of validation in every episode 
    for i in range(1, config.EPISODE + 1):
        loss += model.update(i)

        if i % 1000 == 0:
            print(f'Saving model (epoch = {i}, average loss = {loss / 1000:.4f})')
            model.save()
            model.save_td(loss / 1000)
            actions = validation(valid, model)
            hists.append(model.action_selections)
            p_val, e_val = WIS_estimator(actions, valid_dataset, id_index_map)
            policy_val.append(p_val)
            expert_val.append(e_val)
            print(f'policy WIS estimator: {p_val:.5f}')
            print(f'expert: {e_val:.5f}')
            loss = 0
    
    animation_action_distribution(model, hists)
    plot_training_loss(model)
    plot_action_distribution(model)
    plot_estimate_value(model, expert_val, policy_val)


def WIS_estimator(actions, expert_data, id_index_map):
    '''
    actions         : policy action (tensor)
    expert_data       : original expert dataset (DataFrame)
    id_index_map    : indexes of each icustayid (dict)
    return ->
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


######################################################################################
# Plot
######################################################################################
def plot_training_loss(model):
    x, y = list(), list()

    fig, ax = plt.subplots()

    ax.plot(model.tds)

    ax.set_xlabel('epoch * 1000')
    ax.set_ylabel('loss')

    ax.set_title('training loss')

    plt.savefig(os.path.join(model.log_dir, 'training loss.png'))


def plot_action_distribution(model):
    fig, ax = plt.subplots()

    ax.hist(range(25), weights=model.action_selections, bins=np.arange(26)-0.5)

    ax.set_xlabel('action index')
    ax.set_ylabel('freq')
    ax.set_xticks(range(0, 25))

    ax.set_title('D3QN action distribution')

    plt.savefig(os.path.join(model.log_dir, 'D3QN valid action distribution.png'))


def animation_action_distribution(model, hists):
    fig, ax = plt.subplots()

    def update(i):
        ax.clear()
        ax.hist(range(25), weights=hists[i], bins=np.arange(26)-0.5)
        ax.set_xlabel('action index')
        ax.set_ylabel('freq')
        ax.set_xticks(range(0, 25))
        ax.set_title(f'action distribution {i}')

    ani = FuncAnimation(fig, update, frames=len(hists), interval=200)
    ani.save(os.path.join(model.log_dir, 'D3QN valid action distribution.gif'), writer='imagemagick')


def plot_estimate_value(model, expert_val, policy_val):
    fig, ax = plt.subplots()

    ax.plot(list(range(len(policy_val))), policy_val)
    ax.plot(list(range(len(expert_val))), expert_val)
    ax.legend(['policy', 'expert'],loc='best')

    ax.set_xlabel('epoch * 1000')
    ax.set_ylabel('estimate value')

    ax.set_title('policy vs expert value')

    plt.savefig(os.path.join(model.log_dir, 'valid estimate value.png'))


def train_parallel(config, env, lr, train_dataset, valid_dataset, valid, id_index_map):
    config.LR = lr / 10000

    path = os.path.join('./log', f'batch_size-{config.BATCH_SIZE} episode-{config.EPISODE} use_pri-{config.USE_PRIORITY_REPLAY} lr-{config.LR} reg_lambda-{config.REG_LAMBDA}')
    if not os.path.exists(path):
        os.mkdir(path)

    model = Model(static_policy=False, env=env, config=config, log_dir=path)

    log_path = os.path.join('./log', f'batch_size-{config.BATCH_SIZE} episode-{config.EPISODE} \
                        use_pri-{config.USE_PRIORITY_REPLAY} lr-{config.LR} reg_lambda-{config.REG_LAMBDA} no Normal')
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    agent_path = os.path.join('./saved_agents', f'batch_size-{config.BATCH_SIZE} episode-{config.EPISODE} \
                        use_pri-{config.USE_PRIORITY_REPLAY} lr-{config.LR} reg_lambda-{config.REG_LAMBDA} no Normal')
    if not os.path.exists(agent_path):
        os.mkdir(agent_path)

    model = Model(static_policy=False, env=env, config=config, log_dir=log_path, agent_dir=agent_path)

    ######################################################################################
    # Training
    ######################################################################################
    add_dataset_to_replay(train_dataset, model)
    training(model, valid, config, valid_dataset, id_index_map)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--hour", type=int, help="hours of one state", dest="hour", default=4)
    parser.add_argument("--batch_size", type=int, help="batch_size", dest="batch_size", default=32)
    parser.add_argument("--episode", type=int, help="episode", dest="episode", default=70000)
    parser.add_argument("--use_pri", type=int, help="use priority replay", dest="use_pri", default=1)
    # parser.add_argument("--lr", type=int, help="learning rate", dest="lr", default=0.0001)
    parser.add_argument("--reg_lambda", type=float, help="regularization term coeficient", dest="reg_lambda", default=5)
    args = parser.parse_args()
    hour = args.hour
    batch_size = args.batch_size
    episode = args.episode
    use_pri = args.use_pri
    # lr = args.lr
    reg_lambda = args.reg_lambda

    ######################################################################################
    # Load Dataset
    ######################################################################################
    dataset_path = "../data/final_dataset/"
    train_dataset = pd.read_csv(os.path.join(dataset_path, f'train_{hour}.csv'))
    valid_dataset = pd.read_csv(os.path.join(dataset_path, f'valid_{hour}.csv'))

    ######################################################################################
    # Hyperparameters
    ######################################################################################
    config = Config()

    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # training loop times
    config.EPISODE = episode

    # algorithm control
    config.USE_PRIORITY_REPLAY = use_pri
            
    # Multi-step returns
    config.N_STEPS = 1

    # misc agent variables
    config.GAMMA = 0.99
    # config.LR = lr / 10000

    config.REG_LAMBDA = reg_lambda

    # memory
    exp_replay_size = 1
    while exp_replay_size < train_dataset.shape[0]:
        exp_replay_size <<= 1

    config.EXP_REPLAY_SIZE = exp_replay_size
    config.BATCH_SIZE = batch_size
    config.PRIORITY_ALPHA = 0.6
    config.PRIORITY_BETA_START = 0.9
    config.PRIORITY_BETA_FRAMES = 30000

    # data logging parameters
    config.ACTION_SELECTION_COUNT_FREQUENCY = 10000

    clip_reward = True

    env = {'num_feats': 49, 'num_actions': 25}

    ######################################################################################
    # Parallel training
    ######################################################################################
    valid, id_index_map = get_valid_dataset(valid_dataset)

    processes = list()
    for lr in range(1, 11):
        processes.append(Process(target=train_parallel, args=(config, env, lr, train_dataset, valid_dataset, valid, id_index_map)))
    for i in range(len(processes)):
        processes[i].start()
    for i in range(len(processes)):
        processes[i].join()