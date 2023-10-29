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

from utils import Config
from agents import DQN, WDQN
from network import DuellingMLP

from torch.utils.tensorboard import SummaryWriter

pd.options.mode.chained_assignment = None

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, help="batch_size", default=128)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-4)
    parser.add_argument("--use_pri", type=int, help="use priority replay", default=0)
    parser.add_argument("--agent", type=str, help="agent type", default="D3QN")
    parser.add_argument("--episode", type=int, help="episode", default=1e8)
    parser.add_argument("--eps_start", type=float, help="epsilon start value", default=1.)
    parser.add_argument("--eps_end", type=float, help="epsilon min value", default=0.01)
    parser.add_argument("--eps_decay", type=float, help="epsilon min value", default=0.995)
    parser.add_argument("--target_update_freq", type=int, help="target Q update frequency", default=50)
    parser.add_argument("--update_per_step", type=int, help="update parameters per step", default=1)
    parser.add_argument("--device", type=str, help="device", default="cpu")
    parser.add_argument("--seed", type=int, help="random seed", default=10)
    args = parser.parse_args()
    return args

class D3QN_Agent(DQN):
    def __init__(self, static_policy=False, env=None, config=None, log_dir='./logs'):
        super().__init__(static_policy, env, config, log_dir)

    def declare_networks(self):
        self.model = DuellingMLP(self.num_feats, self.num_actions).to(self.device)
        self.target_model = DuellingMLP(self.num_feats, self.num_actions).to(self.device)

class WD3QN_Agent(WDQN):
    def __init__(self, static_policy=False, env=None, config=None, log_dir='./logs'):
        super().__init__(static_policy, env, config, log_dir)

    def declare_networks(self):
        self.model = DuellingMLP(self.num_feats, self.num_actions).to(self.device)
        self.target_model = DuellingMLP(self.num_feats, self.num_actions).to(self.device)


def training(model: DQN, config: Config, args):
    writer = SummaryWriter(model.log_dir)
    count = 0 # count how many consecutive episode that reward is above 200
    step = 0
    eps = args.eps_start
    for i in range(int(config.EPISODE)):
        ep_reward = 0
        state = env.reset()
        done = False
        t = 0
        while not done and t < 9999:
            action = model.get_action(state, eps)
            next_state, reward, done, _ = env.step(action)
            model.append_to_replay(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            step += 1
            t += 1
            if step % args.update_per_step == 0 and step > config.BATCH_SIZE:
                loss = model.update(i)
                writer.add_scalars('loss', loss, step)
        eps = max(args.eps_end, args.eps_decay * eps)
        writer.add_scalar('ep_reward', ep_reward, i)
        writer.add_scalar('epsilon', eps, i)
        print(f'[EPISODE {i}] | episode reward : {ep_reward}')
        if ep_reward > 200:
            count += 1
            if count > 100:
                model.save()
                break
        else:
            count = 0


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

    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.EPISODE = args.episode
    config.LR = args.lr
    config.BATCH_SIZE = args.batch_size
    config.TARGET_NET_UPDATE_FREQ = args.target_update_freq
    config.USE_PRIORITY_REPLAY = args.use_pri

    env_spec = {'num_feats': 8, 'num_actions': 4}

    path = f'{args.agent}/online_batch_size={config.BATCH_SIZE}-lr={config.LR}-use_pri={config.USE_PRIORITY_REPLAY}'
    log_path = os.path.join('./logs', path)

    if args.agent == 'D3QN':
        model = D3QN_Agent(log_dir=log_path, env=env_spec, config=config)
    elif args.agent == 'WD3QN':
        model = WD3QN_Agent(log_dir=log_path, env=env_spec, config=config)
    else:
        raise NotImplementedError

    os.makedirs(log_path, exist_ok=True)

    training(model, config, args)
