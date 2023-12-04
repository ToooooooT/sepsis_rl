######################################################################################
# Import Package
######################################################################################
import sys
sys.path.append('../')
import numpy as np
import pandas as pd
import torch
import os
import random
from argparse import ArgumentParser
import gym
import pickle

from utils import Config
from agents import DQN, WDQN, SAC, SAC_BC, BaseAgent
from network import DuellingMLP, PolicyMLP
from ope import FQE

pd.options.mode.chained_assignment = None

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, help="batch_size", default=256)
    parser.add_argument("--fqe_batch_size", type=int, help="batch_size", default=256)
    parser.add_argument("--lr", type=float, help="learning rate", default=3e-4)
    parser.add_argument("--fqe_lr", type=float, help="learning rate", default=1e-4)
    parser.add_argument("--use_pri", type=int, help="use priority replay", default=0)
    parser.add_argument("--agent", type=str, help="agent type", default="D3QN")
    parser.add_argument("--episode", type=int, help="episode", default=1e6)
    parser.add_argument("--fqe_episode", type=int, help="episode", default=150)
    parser.add_argument("--test_freq", type=int, help="test frequency", default=10000)
    parser.add_argument("--cpu", action="store_true", help="use cpu")
    parser.add_argument("--clip_expected_return", type=float, help="the value of clipping expected return", default=np.inf)
    parser.add_argument("--gradient_clip", action="store_true", help="gradient clipping in range (-1, 1)")
    parser.add_argument("--dataset", type=str, help="dataset mode", default='train')
    parser.add_argument("--seed", type=int, help="random seed", default=10)
    parser.add_argument("--num_worker", type=int, help="number of worker to handle data loader", default=4)
    parser.add_argument("--load_checkpoint", action="store_true", help="load checkpoint")
    args = parser.parse_args()
    return args

hidden_size = (128, 128)

class D3QN_Agent(DQN):
    def __init__(self, env=None, config=None, log_dir='./logs', static_policy=False):
        super().__init__(env, config, log_dir, static_policy)

    def declare_networks(self):
        self.model = DuellingMLP(self.num_feats, self.num_actions, hidden_size=hidden_size).to(self.device)
        self.target_model = DuellingMLP(self.num_feats, self.num_actions, hidden_size=hidden_size).to(self.device)

class WD3QN_Agent(WDQN):
    def __init__(self, env=None, config=None, log_dir='./logs', static_policy=False):
        super().__init__(env, config, log_dir, static_policy)

    def declare_networks(self):
        self.model = DuellingMLP(self.num_feats, self.num_actions, hidden_size=hidden_size).to(self.device)
        self.target_model = DuellingMLP(self.num_feats, self.num_actions, hidden_size=hidden_size).to(self.device)

class SAC_Agent(SAC):
    def __init__(self, env=None, config=None, log_dir='./logs', static_policy=False):
        super().__init__(env, config, log_dir, static_policy)

    def declare_networks(self):
        self.actor = PolicyMLP(self.num_feats, self.num_actions, hidden_size=hidden_size).to(self.device)
        self.qf1 = DuellingMLP(self.num_feats, self.num_actions, hidden_size=hidden_size).to(self.device)
        self.qf2 = DuellingMLP(self.num_feats, self.num_actions, hidden_size=hidden_size).to(self.device)
        self.target_qf1 = DuellingMLP(self.num_feats, self.num_actions, hidden_size=hidden_size).to(self.device)
        self.target_qf2 = DuellingMLP(self.num_feats, self.num_actions, hidden_size=hidden_size).to(self.device)

class SAC_BC_Agent(SAC_BC):
    def __init__(self, env=None, config=None, log_dir='./logs', static_policy=False):
        super().__init__(env, config, log_dir, static_policy)

    def declare_networks(self):
        self.actor = PolicyMLP(self.num_feats, self.num_actions, hidden_size=hidden_size).to(self.device)
        self.qf1 = DuellingMLP(self.num_feats, self.num_actions, hidden_size=hidden_size).to(self.device)
        self.qf2 = DuellingMLP(self.num_feats, self.num_actions, hidden_size=hidden_size).to(self.device)
        self.target_qf1 = DuellingMLP(self.num_feats, self.num_actions, hidden_size=hidden_size).to(self.device)
        self.target_qf2 = DuellingMLP(self.num_feats, self.num_actions, hidden_size=hidden_size).to(self.device)

def get_agent(args, log_path, env_spec, config):
    if args.agent == 'D3QN':
        model = D3QN_Agent(log_dir=log_path, env=env_spec, config=config)
    elif args.agent == 'WD3QN':
        model = WD3QN_Agent(log_dir=log_path, env=env_spec, config=config)
    elif args.agent == 'SAC':
        model = SAC_Agent(log_dir=log_path, env=env_spec, config=config)
    elif args.agent == 'SAC_BC':
        model = SAC_BC_Agent(log_dir=log_path, env=env_spec, config=config)
    else:
        raise NotImplementedError
    return model

def get_dataset_path(mode):
    pre = './dataset/'
    if mode == "train":
        f = 'train.pkl'
    elif mode == "other":
        f = 'others_train.pkl'
    elif mode == "expert":
        f = 'expert.pkl'
    elif mode == "medium":
        f = 'medium.pkl'
    else:
        raise NotImplementedError
    return os.path.join(pre, f)


def training(agent: DQN, train_dict: dict, test_dict: dict, config: Config, args):
    max_avg_reward = 0
    fqe = FQE(agent, 
              train_dict, 
              test_dict, 
              config, 
              args,
              Q=DuellingMLP(agent.num_feats, agent.num_actions, hidden_size=hidden_size),
              target_Q=DuellingMLP(agent.num_feats, agent.num_actions, hidden_size=hidden_size))

    if args.load_checkpoint:
        start = agent.load_checkpoint()
    else:
        start = -1

    for i in range(start + 1, int(config.EPISODE)):
        loss = agent.update(i)
        if i % args.test_freq == 0:
            avg_reward = testing(agent, 20)

            fqe_return, _ = fqe.estimate(agent=agent)

            with open(os.path.join(agent.log_dir, "expected_return.txt"), "a") as f:
                f.write(f'[EPISODE {i}] | true average reward : {avg_reward}, FQE average reward : {fqe_return} | loss : {loss}\n')
            print(f'[EPISODE {i}] | true average reward : {avg_reward}, FQE average reward : {fqe_return}')
            fqe.records2csv()

            if avg_reward > max_avg_reward:
                max_avg_reward = avg_reward
                agent.save()

        agent.save_checkpoint(i)


def get_actions_probs(test_dict: dict, agent: BaseAgent):
    '''
    Returns:
        action_probs: np.ndarray; expected shape (B, D)
    '''
    states = torch.tensor(test_dict['s'], device=agent.device, dtype=torch.float).view(-1, agent.num_feats)

    with torch.no_grad():
        actions, _, _, action_probs = agent.get_action_probs(states)
        actions = actions.numpy()
        action_probs = action_probs.numpy()

    return action_probs


def testing(model: DQN, episode=10):
    """
        Test the learned model (no change needed)
    """     
    # render = True
    max_episode_len = 10000
    total_reward = 0
    for _ in range(1, episode + 1):
        state = env.reset()
        running_reward = 0
        for _ in range(max_episode_len + 1):
            action = model.get_action(state)
            state, reward, done, _ = env.step(action)
            running_reward += reward
            if done:
                break
        total_reward += running_reward
    return total_reward / episode

def split_dataset(args):
    file_path = get_dataset_path(args.dataset)
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    states = data[0]
    actions = data[1]
    rewards = data[2]
    next_states = data[3]
    dones = data[4]
    train_step = 800000 # 4 : 1
    train_data = [states[:train_step], 
                  actions[:train_step],
                  rewards[:train_step],
                  next_states[:train_step],
                  dones[:train_step]]

    train_dict = {'s': states[:train_step],
                 'a': actions[:train_step],
                 'r': rewards[:train_step],
                 's_': next_states[:train_step],
                 'done': dones[:train_step]}

    done_indexs = np.where(dones == 1)[0]
    start_index = done_indexs[np.where(done_indexs > train_step)[0][0]] + 1
    end_index = done_indexs[-1]
    test_dict = {'s': states[start_index:end_index + 1],
                 'a': actions[start_index:end_index + 1],
                 'r': rewards[start_index:end_index + 1],
                 's_': next_states[start_index:end_index + 1],
                 'done': dones[start_index:end_index + 1]}
    return train_data, train_dict, test_dict


if __name__ == '__main__':
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = gym.make('LunarLander-v2')
    env.reset(seed=args.seed)  

    dataset_path = './dataset'

    ######################################################################################
    # Hyperparameters
    ######################################################################################
    config = Config()

    if args.cpu:
        config.DEVICE = torch.device("cpu")
    else:
        config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {config.DEVICE}")

    config.EPISODE = args.episode
    config.LR = args.lr
    config.BATCH_SIZE = args.batch_size
    config.USE_PRIORITY_REPLAY = args.use_pri
    config.IS_GRADIENT_CLIP = args.gradient_clip

    env_spec = {'num_feats': 8, 'num_actions': 4}

    path = f'{args.agent}/offline-dataset={args.dataset}-batch_size={config.BATCH_SIZE}-lr={config.LR}-use_pri={config.USE_PRIORITY_REPLAY}-hidden_size={hidden_size}'
    log_path = os.path.join('./logs', path)

    agent = get_agent(args, log_path, env_spec, config)

    os.makedirs(log_path, exist_ok=True)
    # load dataset
    train_data, train_dict, test_dict = split_dataset(args)
    agent.memory.read_data(train_data)
    training(agent, train_dict, test_dict, config, args)
