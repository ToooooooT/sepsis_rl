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
import pickle
from argparse import ArgumentParser
from tqdm import tqdm
import gym

from utils import Config
from WD3QNE import Dist_DQN

from torch.utils.tensorboard import SummaryWriter

pd.options.mode.chained_assignment = None

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, help="batch_size", default=32)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-4)
    parser.add_argument("--episode", type=int, help="episode", default=1000000)
    parser.add_argument("--valid_freq", type=int, help="validation frequency", default=1)
    parser.add_argument("--device", type=str, help="device", default="cpu")
    parser.add_argument("--seed", type=int, help="random seed", default=10)
    parser.add_argument("--num_worker", type=int, help="number of worker to handle data loader", default=20)
    args = parser.parse_args()
    return args


def training(model: Dist_DQN, train_data: dict, config, args):
    '''
    Args:
        train_data      : processed training dataset
    '''
    writer = SummaryWriter(model.log_dir)
    loss = 0
    valid_freq = args.valid_freq

    batchs = (torch.tensor(train_data['s'], device=args.device, dtype=torch.float), 
              torch.tensor(train_data['s_'], device=args.device, dtype=torch.float), 
              torch.tensor(train_data['a'], device=args.device, dtype=torch.float), 
              torch.tensor(train_data['r'], device=args.device, dtype=torch.float), 
              torch.tensor(train_data['done'], device=args.device, dtype=torch.float))
    max_avg_reward = 0
    for i in range(config.EPISODE):
        loss = model.train_no_expertise(batchs)
        writer.add_scalar('loss', loss, i)

        if i % valid_freq == 0:
            avg_reward = testing(model)
            writer.add_scalar('test average reward', avg_reward, i)
            print(f'[EPISODE {i}] | test average reward {avg_reward}')

            if avg_reward > max_avg_reward:
                model.save()
                max_avg_reward = avg_reward

                if max_avg_reward > 200:
                    break

def testing(model: Dist_DQN, episode=10):
    """
        Test the learned model (no change needed)
    """     
    # render = True
    max_episode_len = 10000
    total_reward = 0
    for i_episode in range(1, episode + 1):
        state = env.reset()
        running_reward = 0
        for _ in range(max_episode_len + 1):
            state = torch.tensor(state).view(1, -1)
            action = model.get_action(state)[0][0][0]
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
    env.seed(args.seed)  

    ######################################################################################
    # Load Dataset
    ######################################################################################
    '''
    train / valid / test dataset are original unnomalized dataset, with action and reward
    train / valid / test data contain (s, a, r, s_, done, SOFA, is_alive) transitions, with normalization
    '''
    dataset_path = "./dataset"
    # train
    with open(os.path.join(dataset_path, 'medium.pkl'), 'rb') as file:
        train_data = pickle.load(file)

    # test
    # with open(os.path.join(dataset_path, 'test.pkl'), 'rb') as file:
    #     test_data = pickle.load(file)

    ######################################################################################
    # Hyperparameters
    ######################################################################################
    config = Config()

    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.EPISODE = args.episode
    config.LR = args.lr
    config.BATCH_SIZE = args.batch_size

    env_spec = {'num_feats': 8, 'num_actions': 4}

    path = f'WD3QNE/offline_batch_size={config.BATCH_SIZE}-lr={config.LR}'
    log_path = os.path.join('./logs', path)
    os.makedirs(log_path, exist_ok=True)

    model = Dist_DQN(log_dir=log_path, state_dim=env_spec['num_feats'], num_actions=env_spec['num_actions'], lr=config.LR, batch_size=config.BATCH_SIZE)

    ######################################################################################
    # Training
    ######################################################################################
    print('Start training...')
    training(model, train_data, config, args)

    ######################################################################################
    # Testing
    ######################################################################################
    model.load()

    print('Start testing...')
    avg_reward = testing(model)
    print(f'{avg_reward = }')
