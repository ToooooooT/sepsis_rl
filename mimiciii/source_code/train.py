######################################################################################
# Import Package
######################################################################################
import numpy as np
import pandas as pd
import torch
import os
import random
import pickle
from argparse import ArgumentParser
from tqdm import tqdm

from agents import DQN, SAC, BaseAgent
from utils import Config, plot_action_dist, plot_action_distribution, plot_estimate_value, \
                animation_action_distribution, plot_pos_neg_action_dist, plot_diff_action_SOFA_dist, \
                plot_diff_action, plot_survival_rate, plot_expected_return_distribution, \
                WIS_estimator, DR_estimator, plot_action_diff_survival_rate
from network import DuellingMLP, PolicyMLP

from torch.utils.tensorboard import SummaryWriter

pd.options.mode.chained_assignment = None

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--hour", type=int, help="hours of one state", default=4)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=32)
    parser.add_argument("--episode", type=int, help="episode", default=100000)
    parser.add_argument("--use_pri", type=int, help="use priority replay", default=1)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-4)
    parser.add_argument("--reg_lambda", type=int, help="regularization term coeficient", default=5)
    parser.add_argument("--agent", type=str, help="agent type", default="D3QN")
    parser.add_argument("--clip_expected_return", type=int, help="the value of clipping expected return", default=40)
    parser.add_argument("--test_dataset", type=str, help="test dataset", default="test")
    parser.add_argument("--valid_freq", type=int, help="validation frequency", default=50)
    parser.add_argument("--gif_freq", type=int, help="frequency of making validation action distribution gif", default=1000)
    parser.add_argument("--target_net_freq", type=int, help="the frequency of updates for the target networks", default=1)
    parser.add_argument("--env_model_path", type=str, help="path of environment model", default="./log/Env/batch_size=32-lr=0.001-episode=200/model.pth")
    parser.add_argument("--clf_model_path", type=str, help="path of classifier model", default="./log/Clf/LG_clf.sav")
    parser.add_argument("--device", type=str, help="device", default="cpu")
    parser.add_argument("--seed", type=int, help="random seed", default=10)
    parser.add_argument("--num_worker", type=int, help="number of worker to handle data loader", default=20)
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


def add_dataset_to_replay(train_data, model, clip_reward):
    # put all transitions in replay buffer
    s, a, r, s_, SOFA = train_data['s'], train_data['a'], train_data['r'], train_data['s_'], train_data['SOFA']
    if clip_reward:
        r[(r > 1) & (r != 15)] = 1
        r[(r < -1) & (r != -15)] = -1
    for i in range(s.shape[0]):
        model.append_to_replay(s[i], a[i], r[i], s_[i], SOFA[i])

def training(model: BaseAgent, valid_dataset: pd.DataFrame, valid_dict: dict, config, args):
    '''
    Args:
        train_data      : processed training dataset
        valid_dataset   : original valid dataset (DataFrame)
        valud_dict      : processed validation dataset
    '''
    writer = SummaryWriter(model.log_dir)
    loss = 0
    avg_wis_policy_returns = list()
    avg_dr_policy_returns = list()
    hists = list() # save model actions of validation in every episode 
    valid_freq = args.valid_freq
    gif_freq = args.gif_freq
    max_expected_return = -np.inf
    valid_data = valid_dict['data']
    valid_id_index_map = valid_dict['id_index_map']
    dre = DR_estimator(valid_dataset, valid_dict, args, model.device)

    for i in tqdm(range(1, config.EPISODE + 1)):
        loss = model.update(i)
        writer.add_scalars('loss', loss, i)

        if i % valid_freq == 0:
            actions, action_probs, est_q_values = testing(valid_data, model)
            # store actions
            if i % gif_freq == 0:
                hists.append(model.action_selections)
            # estimate expected return
            avg_wis_p_return, _ = WIS_estimator(action_probs, valid_dataset, valid_id_index_map, args.clip_expected_return)
            avg_wis_policy_returns.append(avg_wis_p_return)
            avg_dr_p_return, _, _ = dre.estimate_expected_return(est_q_values, actions, action_probs, valid_dataset, valid_id_index_map)
            avg_dr_policy_returns.append(avg_dr_p_return)
            if args.agent == 'D3QN':
                if avg_dr_p_return > max_expected_return:
                    max_expected_return = avg_dr_p_return
                    model.save()
            elif args.agent == 'SAC':
                if avg_wis_p_return > max_expected_return:
                    max_expected_return = avg_wis_p_return
                    model.save()
            # record in tensorboard
            writer.add_scalars('expected return validation', \
                               dict(zip(['WIS', 'DR'], [avg_wis_p_return, avg_dr_p_return])), i)

    plot_action_distribution(model.action_selections, model.log_dir)
    animation_action_distribution(hists, model.log_dir)
    avg_wis_policy_returns = np.array(avg_wis_policy_returns)
    avg_dr_policy_returns = np.array(avg_dr_policy_returns)
    plot_estimate_value(np.vstack((avg_wis_policy_returns, avg_dr_policy_returns)), ['WIS', 'DR'], model.log_dir, valid_freq)


def testing(test_data, model: BaseAgent):
    '''
    Returns:
        actions     : np.ndarray; expected shape (B, 1)
        action_probs: np.ndarray; expected shape (B, D)
        est_q_values: np.ndarray; expected shape (B, 1)
    '''
    batch_state, batch_action = test_data['s'], test_data['a']

    batch_state = torch.tensor(batch_state, device=model.device, dtype=torch.float).view(-1, model.num_feats)
    batch_action = torch.tensor(batch_action, device=model.device, dtype=torch.int64).view(-1, 1)

    with torch.no_grad():
        if isinstance(model, D3QN_Agent):
            model.model.eval()
            est_q_values, actions = model.model(batch_state).max(dim=1)
            actions = actions.view(-1, 1).detach().cpu().numpy() # (B, 1)
            est_q_values = est_q_values.view(-1, 1).detach().cpu().numpy() # (B, 1)
            # assume policy take the max action in probability of 0.99 and any other actions of 0.01 
            action_probs = np.full((actions.shape[0], 25), 0.01)
            action_probs[np.arange(actions.shape[0]), actions[:, 0]] = 0.99
        elif isinstance(model, SAC_Agent):
            model.actor.eval()
            actions, _, _, action_probs = model.actor.get_action(batch_state)
            actions = actions.view(-1, 1).detach().cpu().numpy() # (B, 1)
            action_probs = action_probs.detach().cpu().numpy() # (B, D)
            # TODO: est_q_values
            est_q_values = np.zeros((actions.shape[0], 1))

    model.save_action(actions)

    return actions, action_probs, est_q_values


if __name__ == '__main__':
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ######################################################################################
    # Load Dataset
    ######################################################################################
    '''
    train / valid / test dataset are original unnomalized dataset, with action and reward
    train / valid / test data contain (s, a, r, s_, done, SOFA, is_alive) transitions, with normalization
    '''
    dataset_path = "../data/final_dataset/"

    # train
    train_dataset = pd.read_csv(os.path.join(dataset_path, f'train_{args.hour}.csv'))
    icustayids = train_dataset['icustayid'].unique()
    with open(os.path.join(dataset_path, 'train.pkl'), 'rb') as file:
        train_dict = pickle.load(file)
    train_data = train_dict['data']

    # validation
    valid_dataset = pd.read_csv(os.path.join(dataset_path, f'valid_{args.hour}.csv'))
    with open(os.path.join(dataset_path, 'valid.pkl'), 'rb') as file:
        valid_dict = pickle.load(file)

    # test
    test_dataset = pd.read_csv(os.path.join(dataset_path, f'test_{args.hour}.csv'))
    with open(os.path.join(dataset_path, 'test.pkl'), 'rb') as file:
        test_dict = pickle.load(file)
    test_data, test_id_index_map = test_dict['data'], test_dict['id_index_map']

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

    if args.agent == 'D3QN':
        path = f'D3QN/episode={config.EPISODE}-batch_size={config.BATCH_SIZE}-use_pri={config.USE_PRIORITY_REPLAY}-lr={config.LR}-reg_lambda={config.REG_LAMBDA}-target_net_freq={config.TARGET_NET_UPDATE_FREQ}'
    else:
        path = f'SAC/episode={config.EPISODE}-batch_size={config.BATCH_SIZE}-use_pri={config.USE_PRIORITY_REPLAY}-lr={config.LR}-target_net_freq={config.TARGET_NET_UPDATE_FREQ}'
    log_path = os.path.join('./log', path)
    os.makedirs(log_path, exist_ok=True)

    if args.agent == 'D3QN':
        model = D3QN_Agent(static_policy=False, env=env, config=config, log_dir=log_path)
    elif args.agent == 'SAC':
        model = SAC_Agent(static_policy=False, env=env, config=config, log_dir=log_path)
    else:
        raise NotImplementedError

    ######################################################################################
    # Training
    ######################################################################################
    print('Adding dataset to replay buffer...')
    add_dataset_to_replay(train_data, model, clip_reward)

    print('Start training...')
    training(model, valid_dataset, valid_dict, config, args)

    ######################################################################################
    # Testing
    ######################################################################################
    model.load()

    print('Start testing...')
    actions, action_probs, est_q_values = testing(test_data, model)

    test_dataset['policy action'] = actions
    test_dataset['policy iv'] = actions / 5
    test_dataset['policy vaso'] = actions % 5
    # estimate expected return
    avg_wis_policy_return, wis_policy_return = WIS_estimator(action_probs, test_dataset, test_id_index_map, args.clip_expected_return)
    dre = DR_estimator(test_dataset, test_dict, args, config.device)
    avg_dr_policy_return, dr_policy_return, est_alive = \
        dre.estimate_expected_return(est_q_values, actions, action_probs, test_dataset, test_id_index_map)
    # plot expected return result
    policy_returns = np.vstack((wis_policy_return, dr_policy_return))
    plot_expected_return_distribution(policy_returns, ['WIS', 'DR'], log_path)
    plot_survival_rate(policy_returns, test_id_index_map, test_dataset, ['WIS', 'DR'], log_path)
    # plot action distribution
    negative_traj = test_dataset.query('died_in_hosp == 1.0 | died_within_48h_of_out_time == 1.0 | mortality_90d == 1.0')
    positive_traj = test_dataset.query('died_in_hosp != 1.0 & died_within_48h_of_out_time != 1.0 & mortality_90d != 1.0')
    plot_action_dist(actions, test_dataset, log_path)
    plot_pos_neg_action_dist(positive_traj, negative_traj, log_path)
    plot_diff_action_SOFA_dist(positive_traj, negative_traj, log_path)
    plot_diff_action(positive_traj, negative_traj, log_path)
    plot_action_diff_survival_rate(train_dataset, test_dataset, log_path)
    # store result in text file
    with open(os.path.join(log_path, 'evaluation.txt'), 'w') as f:
        f.write(f'policy WIS estimator: {avg_wis_policy_return:.5f}\n')
        f.write(f'policy DR estimator: {avg_dr_policy_return:.5f}\n')
        f.write(f'Logistic regression survival rate: {est_alive.mean():.5f}\n')
    # print result
    print(f'policy WIS estimator: {avg_wis_policy_return:.5f}')
    print(f'policy DR estimator: {avg_dr_policy_return:.5f}')
    print(f'Logistic regression survival rate: {est_alive.mean():.5f}')

    test_dataset.to_csv(os.path.join(model.log_dir, 'test_data_predict.csv'), index=False)