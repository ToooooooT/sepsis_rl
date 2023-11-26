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
from agents import DQN, WDQN, SAC
from network import DuellingMLP, PolicyMLP

from torch.utils.tensorboard import SummaryWriter

pd.options.mode.chained_assignment = None

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, help="batch_size", default=128)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-4)
    parser.add_argument("--use_pri", type=int, help="use priority replay", default=0)
    parser.add_argument("--agent", type=str, help="agent type", default="D3QN")
    parser.add_argument("--step", type=int, help="episode", default=1e6)
    parser.add_argument("--eps_start", type=float, help="epsilon start value", default=1.)
    parser.add_argument("--eps_end", type=float, help="epsilon min value", default=0.01)
    parser.add_argument("--eps_decay", type=float, help="epsilon min value", default=0.995)
    parser.add_argument("--update_per_step", type=int, help="update parameters per step", default=1)
    parser.add_argument("--save_path", type=str, help="save file path of dataset", default="./dataset/train.pkl")
    parser.add_argument("--cpu", action="store_true", help="use cpu")
    parser.add_argument("--seed", type=int, help="random seed", default=10)
    args = parser.parse_args()
    return args

class D3QN_Agent(DQN):
    def __init__(self, env=None, config=None, log_dir='./logs', static_policy=False):
        super().__init__(env, config, log_dir, static_policy)

    def declare_networks(self):
        self.model = DuellingMLP(self.num_feats, self.num_actions).to(self.device)
        self.target_model = DuellingMLP(self.num_feats, self.num_actions).to(self.device)

class WD3QN_Agent(WDQN):
    def __init__(self, env=None, config=None, log_dir='./logs', static_policy=False):
        super().__init__(env, config, log_dir, static_policy)

    def declare_networks(self):
        self.model = DuellingMLP(self.num_feats, self.num_actions).to(self.device)
        self.target_model = DuellingMLP(self.num_feats, self.num_actions).to(self.device)

class SAC_Agent(SAC):
    def __init__(self, env=None, config=None, log_dir='./logs', static_policy=False):
        super().__init__(env, config, log_dir, static_policy)

    def declare_networks(self):
        self.actor = PolicyMLP(self.num_feats, self.num_actions).to(self.device)
        self.qf1 = DuellingMLP(self.num_feats, self.num_actions).to(self.device)
        self.qf2 = DuellingMLP(self.num_feats, self.num_actions).to(self.device)
        self.target_qf1 = DuellingMLP(self.num_feats, self.num_actions).to(self.device)
        self.target_qf2 = DuellingMLP(self.num_feats, self.num_actions).to(self.device)

def get_agent(args, log_path, env_spec, config):
    if args.agent == 'D3QN':
        model = D3QN_Agent(log_dir=log_path, env=env_spec, config=config)
    elif args.agent == 'WD3QN':
        model = WD3QN_Agent(log_dir=log_path, env=env_spec, config=config)
    elif args.agent == 'SAC':
        model = SAC_Agent(log_dir=log_path, env=env_spec, config=config)
    else:
        raise NotImplementedError
    return model


def training(model: DQN, config: Config, args):
    writer = SummaryWriter(model.log_dir)
    step = 0
    eps = args.eps_start
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    while step < args.step:
        ep_reward = 0
        state = env.reset()
        done = False
        t = 0
        while not done and t < 9999:
            action = model.get_action(state, eps)
            next_state, reward, done, _ = env.step(action)
            model.append_to_replay(state, action, reward, next_state, done)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            state = next_state
            ep_reward += reward
            step += 1
            t += 1
            if step % args.update_per_step == 0 and step > config.BATCH_SIZE:
                model.update(step)
            if step == args.step:
                break
        eps = max(args.eps_end, args.eps_decay * eps)
        writer.add_scalar('ep_reward', ep_reward, step)
        print(f'[STEP {step}] | episode reward : {ep_reward}')

    states = np.array(states)
    actions = np.array(actions).reshape(-1, 1)
    rewards = np.array(rewards).reshape(-1, 1)
    next_states = np.array(next_states)
    dones = np.array(dones).reshape(-1, 1)
    data = [states, actions, rewards, next_states, dones]
    with open(args.save_path, "wb") as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = gym.make('LunarLander-v2')
    env.reset(seed=args.seed)  

    ######################################################################################
    # Hyperparameters
    ######################################################################################
    config = Config()

    if args.cpu:
        config.device = torch.device("cpu")
    else:
        config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.LR = args.lr
    config.BATCH_SIZE = args.batch_size
    config.USE_PRIORITY_REPLAY = args.use_pri

    env_spec = {'num_feats': 8, 'num_actions': 4}

    path = f'{args.agent}/generator_batch_size={config.BATCH_SIZE}-lr={config.LR}-use_pri={config.USE_PRIORITY_REPLAY}-hidden_size={(128, 128)}'
    log_path = os.path.join('./logs', path)

    model = get_agent(args, log_path, env_spec, config)

    os.makedirs(log_path, exist_ok=True)

    training(model, config, args)
