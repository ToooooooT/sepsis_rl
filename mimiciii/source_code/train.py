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

from agents import DQN_regularization, WDQNE, SAC_BC_E
from utils import Config, plot_action_dist, plot_estimate_value, \
                animation_action_distribution, plot_pos_neg_action_dist, plot_diff_action_SOFA_dist, \
                plot_diff_action, plot_survival_rate, plot_expected_return_distribution, \
                plot_action_diff_survival_rate
from network import DuellingMLP, PolicyMLP
from ope import WIS, DoublyRobust, q_value_estimator

from torch.utils.tensorboard import SummaryWriter

pd.options.mode.chained_assignment = None

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--hour", type=int, help="hours of one state", default=4)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=128)
    parser.add_argument("--episode", type=int, help="episode", default=1e6)
    parser.add_argument("--use_pri", type=int, help="use priority replay", default=1)
    parser.add_argument("--lr", type=float, help="learning rate", default=3e-4)
    parser.add_argument("--reg_lambda", type=int, help="regularization term coeficient", default=5)
    parser.add_argument("--agent", type=str, help="agent type", default="D3QN")
    parser.add_argument("--clip_expected_return", type=float, help="the value of clipping expected return", default=np.inf)
    parser.add_argument("--test_dataset", type=str, help="test dataset", default="test")
    parser.add_argument("--valid_freq", type=int, help="validation frequency", default=500)
    parser.add_argument("--gif_freq", type=int, help="frequency of making validation action distribution gif", default=1000)
    parser.add_argument("--env_model_path", type=str, help="path of environment model", default="./logs/Env/batch_size=32-lr=0.001-episode=200/model.pth")
    parser.add_argument("--clf_model_path", type=str, help="path of classifier model", default="./logs/Clf/LG_clf.sav")
    parser.add_argument("--cpu", action="store_true", help="use cpu")
    parser.add_argument("--gradient_clip", action="store_true", help="gradient clipping in range (-1, 1)")
    parser.add_argument("--seed", type=int, help="random seed", default=10)
    parser.add_argument("--num_worker", type=int, help="number of worker to handle data loader", default=20)
    args = parser.parse_args()
    return args

hidden_size = (128, 128)

######################################################################################
# Agent
######################################################################################
class D3QN_Agent(DQN_regularization):
    def __init__(self, 
                 env: dict, 
                 config: Config, 
                 log_dir='./log',
                 static_policy=False):
        super().__init__(env, config, log_dir, static_policy)

    def declare_networks(self):
        self.model = DuellingMLP(self.num_feats, self.num_actions, hidden_size=hidden_size).to(self.device)
        self.target_model = DuellingMLP(self.num_feats, self.num_actions, hidden_size=hidden_size).to(self.device)

class WD3QNE_Agent(WDQNE):
    def __init__(self, 
                 env: dict, 
                 config: Config, 
                 log_dir='./log',
                 static_policy=False):
        super().__init__(env, config, log_dir, static_policy)

    def declare_networks(self):
        self.model = DuellingMLP(self.num_feats, self.num_actions, hidden_size=hidden_size).to(self.device)
        self.target_model = DuellingMLP(self.num_feats, self.num_actions, hidden_size=hidden_size).to(self.device)

class SAC_BC_Agent(SAC_BC_E):
    def __init__(self, 
                 env: dict, 
                 config: Config, 
                 log_dir='./log',
                 static_policy=False):
        super().__init__(env, config, log_dir, static_policy)

    def declare_networks(self):
        self.actor = PolicyMLP(self.num_feats, self.num_actions, hidden_size=hidden_size).to(self.device)
        self.qf1 = DuellingMLP(self.num_feats, self.num_actions, hidden_size=hidden_size).to(self.device)
        self.qf2 = DuellingMLP(self.num_feats, self.num_actions, hidden_size=hidden_size).to(self.device)
        self.target_qf1 = DuellingMLP(self.num_feats, self.num_actions, hidden_size=hidden_size).to(self.device)
        self.target_qf2 = DuellingMLP(self.num_feats, self.num_actions, hidden_size=hidden_size).to(self.device)

def get_agent(args, log_path, env_spec, config):
    if args.agent == 'D3QN':
        agent = D3QN_Agent(log_dir=log_path, env=env_spec, config=config)
    elif args.agent == 'WD3QNE':
        agent = WD3QNE_Agent(log_dir=log_path, env=env_spec, config=config)
    elif args.agent == 'SAC_BC':
        agent = SAC_BC_Agent(log_dir=log_path, env=env_spec, config=config)
    else:
        raise NotImplementedError
    return agent

def add_dataset_to_replay(train_data, agent: DQN_regularization, clip_reward):
    # put all transitions in replay buffer
    s = train_data['s']
    a = train_data['a']
    r = train_data['r']
    s_ = train_data['s_']
    a_ = train_data['a_']
    done = train_data['done']
    SOFA = train_data['SOFA']

    if clip_reward:
        r[(r > 1) & (r != 15)] = 1
        r[(r < -1) & (r != -15)] = -1
    if isinstance(agent, D3QN_Agent):
        data = [s, a, r, s_, done]
        agent.memory.read_data(data)
    elif isinstance(agent, WD3QNE_Agent):
        data = [s, a, r, s_, a_, done, SOFA]
        agent.memory.read_data(data)
    elif isinstance(agent, SAC_BC_Agent):
        data = [s, a, r, s_, done, SOFA]
        agent.memory.read_data(data)
    else:
        raise NotImplementedError

def training(agent: D3QN_Agent, valid_dataset: pd.DataFrame, valid_dict: dict, config: Config, args):
    '''
    Args:
        train_data      : processed training dataset
        valid_dataset   : original valid dataset (DataFrame)
        valud_dict      : processed validation dataset
    '''
    writer = SummaryWriter(agent.log_dir)
    avg_wis_policy_returns = []
    avg_dr_policy_returns = []
    hists = [] # save model actions of validation in every episode 
    valid_freq = args.valid_freq
    gif_freq = args.gif_freq
    max_expected_return = -np.inf
    valid_data = valid_dict['data']
    dr = DoublyRobust(valid_dataset, valid_dict, config, args)
    wis = WIS(valid_dataset, valid_dict, config, args)

    for i in tqdm(range(1, int(config.EPISODE) + 1)):
        loss = agent.update(i)
        writer.add_scalars('loss', loss, i)

        if i % valid_freq == 0:
            actions, action_probs, est_q_values = testing(valid_data, agent)
            # store actions in histogram to show animation
            if i % gif_freq == 0:
                hists.append(np.bincount(actions.reshape(-1), minlength=25))
            # estimate expected return
            avg_wis_p_return, _ = wis.estimate(action_probs=action_probs)
            avg_wis_policy_returns.append(avg_wis_p_return)
            avg_dr_p_return, _, _ = dr.estimate(est_q_values=est_q_values, 
                                                 actions=actions, 
                                                 action_probs=action_probs)
            avg_dr_policy_returns.append(avg_dr_p_return)
            if isinstance(agent, D3QN_Agent) or isinstance(agent, WD3QNE_Agent):
                if avg_dr_p_return > max_expected_return:
                    max_expected_return = avg_dr_p_return
                    agent.save()
            # TODO: implement SAC+BC
            else:
                raise NotImplementedError
            writer.add_scalars('expected return validation', \
                               dict(zip(['WIS', 'DR'], [avg_wis_p_return, avg_dr_p_return])), i)

    animation_action_distribution(hists, agent.log_dir)
    avg_wis_policy_returns = np.array(avg_wis_policy_returns)
    avg_dr_policy_returns = np.array(avg_dr_policy_returns)
    plot_estimate_value(np.vstack((avg_wis_policy_returns, avg_dr_policy_returns)), ['WIS', 'DR'], agent.log_dir, valid_freq)


def testing(test_data, agent: D3QN_Agent):
    '''
    Returns:
        actions     : np.ndarray; expected shape (B, 1)
        action_probs: np.ndarray; expected shape (B, D)
        est_q_values: np.ndarray; expected shape (B, 1)
    '''
    batch_state = torch.tensor(test_data['s'], device=agent.device, dtype=torch.float).view(-1, agent.num_feats)

    with torch.no_grad():
        if isinstance(agent, D3QN_Agent) or isinstance(agent, WD3QNE_Agent):
            agent.model.eval()
            est_q_values, actions = agent.model(batch_state).max(dim=1)
            actions = actions.view(-1, 1).detach().cpu().numpy() # (B, 1)
            est_q_values = est_q_values.view(-1, 1).detach().cpu().numpy() # (B, 1)
            # assume policy take the max action in probability of 0.99 and any other actions of 0.01 
            action_probs = np.full((actions.shape[0], 25), 0.01)
            action_probs[np.arange(actions.shape[0]), actions[:, 0]] = 0.99
        # TODO: implement SAC+BC
        else:
            raise NotImplementedError

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

    if args.cpu:
        config.DEVICE = torch.device("cpu")
    else:
        config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config.EPISODE = args.episode
    config.USE_PRIORITY_REPLAY = args.use_pri
    config.LR = args.lr
    config.BATCH_SIZE = args.batch_size
    config.USE_PRIORITY_REPLAY = args.use_pri
    config.EXP_REPLAY_SIZE = len(train_data['s'])
    config.IS_GRADIENT_CLIP = args.gradient_clip
    config.REG_LAMBDA = args.reg_lambda

    env_spec = {'num_feats': 49, 'num_actions': 25}

    if args.agent == 'D3QN':
        path = f'D3QN/episode={config.EPISODE}-batch_size={config.BATCH_SIZE}-use_pri={config.USE_PRIORITY_REPLAY}-lr={config.LR}-reg_lambda={config.REG_LAMBDA}-hidden_size={hidden_size}'
    else:
        path = f'{args.agent}/episode={config.EPISODE}-batch_size={config.BATCH_SIZE}-use_pri={config.USE_PRIORITY_REPLAY}-lr={config.LR}-hidden_size={hidden_size}'
    log_path = os.path.join('./logs', path)

    agent = get_agent(args, log_path, env_spec, config)

    os.makedirs(log_path, exist_ok=True)

    ######################################################################################
    # Training
    ######################################################################################
    print('Adding dataset to replay buffer...')
    add_dataset_to_replay(train_data, agent, clip_reward=True)

    print('Start training...')
    training(agent, valid_dataset, valid_dict, config, args)

    ######################################################################################
    # Testing
    ######################################################################################
    agent.load()

    print('Start testing...')
    actions, action_probs, est_q_values = testing(test_data, agent)

    test_dataset['policy action'] = actions
    test_dataset['policy iv'] = actions / 5
    test_dataset['policy vaso'] = actions % 5

    # estimate expected return
    wis = WIS(test_dataset, test_dict, config, args)
    avg_wis_policy_return, wis_policy_return = wis.estimate(action_probs=action_probs)
    dre = DoublyRobust(test_dataset, test_dict, config, args)
    avg_dr_policy_return, dr_policy_return, est_alive = dre.estimate(est_q_values=est_q_values, 
                                                                     actions=actions, 
                                                                     action_probs=action_probs)
    # plot expected return result
    policy_returns = np.vstack((wis_policy_return, dr_policy_return))
    plot_expected_return_distribution(policy_returns, ['WIS', 'DR'], log_path)
    plot_survival_rate(policy_returns, test_id_index_map, test_dataset, ['WIS', 'DR'], log_path)

    # plot action distribution
    negative_traj = test_dataset.query('mortality_90d == 1.0')
    positive_traj = test_dataset.query('mortality_90d != 1.0')
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

    test_dataset.to_csv(os.path.join(agent.log_dir, 'test_data_predict.csv'), index=False)
