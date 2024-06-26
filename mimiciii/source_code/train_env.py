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

from network import EnvMLP

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F

pd.options.mode.chained_assignment = None

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--hour", type=int, help="hours of one state", default=4)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=32)
    parser.add_argument("--episode", type=int, help="episode", default=100)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--optimizer", type=str, help="optimizer", default="adam")
    parser.add_argument("--test_dataset", type=str, help="test dataset", default="test")
    parser.add_argument("--valid_freq", type=int, help="validation frequency", default=1)
    parser.add_argument("--hidden_size", type=str, help="the dimension of hidden layer size of environment model", default="500,500")
    parser.add_argument("--num_worker", type=int, help="number of worker to handle data loader", default=32)
    parser.add_argument("--device", type=str, help="device", default="cpu")
    parser.add_argument("--seed", type=int, help="random seed", default=10)
    args = parser.parse_args()
    return args

######################################################################################
# Dataset
######################################################################################
class sepsis_dataset(Dataset):
    def __init__(self, state, action, id_index_map, terminal_index) -> None:
        '''
        Args:
            state: numpy array; expected shape (B, S)
            action: numpy arrat; expected shape (B,)
        '''
        super().__init__()
        self.state_dim = state.shape[1]
        # add a zero in last row to avoid overflow of index in next state
        state = np.vstack((state, np.zeros((1, self.state_dim))))
        self.state = state

        self.action_dim = 2
        # add a zero in last row to avoid overflow of index in next state
        self.action = np.array([(x // 5, x % 5) for x in action])
        self.action = np.vstack((self.action, np.zeros((1, self.action_dim))))

        self.data = np.concatenate([self.state, self.action], axis=1)

        self.id_index_map = id_index_map
        self.terminal_index = terminal_index

    def __len__(self):
        return len(self.action) - 1

    def __getitem__(self, index):
        state = self.state[index]
        next_state = self.state[index + 1]
        state_action_pair = self.data[index]
        done = np.array([int(index in self.terminal_index)]) 
        return torch.tensor(state_action_pair, dtype=torch.float), \
            torch.tensor(next_state - state, dtype=torch.float), \
            torch.tensor(done, dtype=torch.float)


def training(model: EnvMLP, optimizer, input, label, done, device):
    input = input.to(device)
    label = label.to(device)
    done = done.to(device)

    pred = model(input)
    loss = F.mse_loss(pred * (1 - done), label * (1 - done))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.detach().cpu().item()


def validation(model: EnvMLP, input, label, done, device):
    input = input.to(device)
    label = label.to(device)
    done = done.to(device)

    with torch.no_grad():
        pred = model(input)
        loss = F.mse_loss(pred * (1 - done), label * (1 - done))
    return loss.detach().cpu().item()


def testing(model: EnvMLP, input, label, done, device):
    input = input.to(device)
    label = label.to(device)
    done = done.to(device)

    with torch.no_grad():
        pred = model(input)
        loss = F.mse_loss(pred * (1 - done), label * (1 - done))
    return loss.detach().cpu().item(), pred.detach().cpu().numpy() + input[:, :-2].detach().cpu().numpy()


if __name__ == '__main__':
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == 'cuda':
        assert(torch.cuda.is_available())

    ######################################################################################
    # Load Dataset
    ######################################################################################
    dataset_path = "../data/final_dataset/"
    full_data = pd.read_csv(os.path.join(dataset_path, f'dataset_{args.hour}.csv'))
    train_dataset = pd.read_csv(os.path.join(dataset_path, f'train_{args.hour}.csv'))
    valid_dataset = pd.read_csv(os.path.join(dataset_path, f'valid_{args.hour}.csv'))
    test_dataset = pd.read_csv(os.path.join(dataset_path, f'{args.test_dataset}_{args.hour}.csv'))

    with open(os.path.join(dataset_path, 'train.pkl'), 'rb') as file:
        train_dict = pickle.load(file)
    train_data, train_id_index_map, train_terminal_index = train_dict['data'], train_dict['id_index_map'], train_dict['terminal_index']

    with open(os.path.join(dataset_path, 'valid.pkl'), 'rb') as file:
        valid_dict = pickle.load(file)
    valid_data, valid_id_index_map, valid_terminal_index = valid_dict['data'], valid_dict['id_index_map'], valid_dict['terminal_index']

    with open(os.path.join(dataset_path, 'test.pkl'), 'rb') as file:
        test_dict = pickle.load(file)
    test_data, test_id_index_map, test_terminal_index = test_dict['data'], test_dict['id_index_map'], test_dict['terminal_index']

    ######################################################################################
    # Hyperparameters
    ######################################################################################
    clip_reward = True

    env = {'num_feats': 49, 'num_actions': 25}

    path = f'Env/batch_size={args.batch_size}-lr={args.lr}-episode={args.episode}'
    log_path = os.path.join('./log', path)
    os.makedirs(log_path, exist_ok=True)
    hidden_size = [int(x) for x in args.hidden_size.split(',')]
    model = EnvMLP(env['num_feats'], 2, hidden_size=hidden_size).to(args.device)
    writer = SummaryWriter(log_path)
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise ValueError

    ######################################################################################
    # Create Dataloader
    ######################################################################################
    train_loader = DataLoader(sepsis_dataset(train_data['s'], train_data['a'], train_id_index_map, train_terminal_index),
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True,
                              num_workers=args.num_worker)

    valid_loader = DataLoader(sepsis_dataset(valid_data['s'], valid_data['a'], valid_id_index_map, valid_terminal_index),
                              batch_size=args.batch_size,
                              shuffle=False,
                              drop_last=False,
                              pin_memory=True,
                              num_workers=args.num_worker)

    test_loader = DataLoader(sepsis_dataset(test_data['s'], test_data['a'], test_id_index_map, test_terminal_index),
                             batch_size=args.batch_size,
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True,
                             num_workers=args.num_worker)

    ######################################################################################
    # Training
    ######################################################################################
    print('Start training...')
    min_loss = np.Inf
    for epoch in tqdm(range(1, args.episode + 1)):
        model.train()
        epoch_loss = 0
        for input, label, done in train_loader:
            loss = training(model, optimizer, input, label, done, args.device)
            epoch_loss += loss

        with open(os.path.join(log_path, 'train_record.txt'), 'a') as f:
            f.write(f'[epoch: {epoch:05d}] training loss: {epoch_loss:.5f}\n')
        writer.add_scalar('training loss', epoch_loss, epoch)

        if epoch % args.valid_freq == 0:
            model.eval()
            valid_loss = 0
            for input, label, done in valid_loader:
                loss = validation(model, input, label, done, args.device)
                valid_loss += loss

            with open(os.path.join(log_path, 'train_record.txt'), 'a') as f:
                f.write(f'[epoch: {epoch:05d}] validation loss: {valid_loss:.5f}\n')
            writer.add_scalar('validation loss', valid_loss, epoch)

            if valid_loss < min_loss:
                min_loss = valid_loss
                torch.save({
                    'env' : model,
                    }, os.path.join(log_path, 'model.pth')
                )

    ######################################################################################
    # Testing
    ######################################################################################
    model.eval()
    test_loss = 0
    est_next_states_test = []
    for input, label, done in test_loader:
        loss, pred = testing(model, input, label, done, args.device)
        test_loss += loss
        est_next_states_test.append(pred)

    with open(os.path.join(log_path, 'train_record.txt'), 'a') as f:
        f.write(f'testing loss: {test_loss:.5f}\n')
    print(f'testing loss: {test_loss:.5f}')