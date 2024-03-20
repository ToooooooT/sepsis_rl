import numpy as np
import pickle
import os
from argparse import ArgumentParser
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import xgboost as xgb
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import mlflow
import lightgbm as lgb
import random
from tqdm import tqdm

from network.behavior_policy import MLP

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, help="random seed", default=10)
    parser.add_argument("--gen_run_id", type=str, help="run id to load model for generating behavior policy prob", default=None)
    parser.add_argument("--log_dir", type=str, help="log directory", default='./logs/behavior_policy')
    parser.add_argument("--dataset_version", type=str, help="dataset version", default='v1_20849')
    parser.add_argument("--reward_type", type=int, help="reward type of dataset", default=0)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--batch_size", type=int, help="batch size", default=32)
    parser.add_argument("--optimizer", type=str, help="optimizer", default='adam')
    parser.add_argument("--episode", type=int, help="episode", default=3000)
    parser.add_argument("--valid_freq", type=int, help="validation frequenct", default=1)
    parser.add_argument("--model", type=str, help="types of model", default="lg")
    parser.add_argument("--cpu", type=int, help="number of cpu core", default=16)
    parser.add_argument("--num_class", type=int, help="number of class", default=25)
    parser.add_argument("--early_stop_rounds", type=int, help="", default=100)
    parser.add_argument("--max_depth", type=int, help="max depth", default=10)
    parser.add_argument("--num_leaves", type=int, help="number of leaves", default=31)
    parser.add_argument("--subsample", type=float, help="sample ratio of dataset", default=0.8)
    parser.add_argument("--colsample_bytree", type=float, help="randomly select a subset of features on each iteration (tree)", default=0.8)
    parser.add_argument("--subsample_freq", type=int, help="frequency for sample ratio of dataset", default=1)
    args = parser.parse_args()
    return args


def train_xgb(args, 
              train_X: np.ndarray, 
              train_Y: np.ndarray, 
              valid_X: np.ndarray, 
              valid_Y: np.ndarray, 
              test_X: np.ndarray, 
              test_Y: np.ndarray):
    # Create DMatrix for training and validation
    dtrain = xgb.DMatrix(train_X, label=train_Y)
    dvalid = xgb.DMatrix(valid_X, label=valid_Y)

    # Define XGBoost parameters
    params = {
        'objective': 'multi:softmax',  # Change to 'multi:softmax' for multi-class classification
        'eval_metric': 'mlogloss',        # Change to 'mlogloss' for multi-class classification
        'num_class': args.num_class,
        'seed': args.seed,
        'eta': args.lr,
        'max_depth': args.max_depth,
        'min_child_weight': 1,
        'subsample': args.subsample,
        'colsample_bytree': args.colsample_bytree,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    mlflow.log_params(params)

    # Train the model with early stopping
    evals = [(dtrain, 'train'), (dvalid, 'eval')]
    num_rounds = args.episode  # Adjust as needed

    model = xgb.train(params, 
                      dtrain, 
                      num_rounds, 
                      evals=evals, 
                      early_stopping_rounds=args.early_stop_rounds, 
                      verbose_eval=True)
    mlflow.xgboost.log_model(model, 'model')

    # Make predictions on the test set
    y_pred = model.predict(xgb.DMatrix(test_X))

    # Evaluate the model
    accuracy = accuracy_score(test_Y, y_pred)
    result = f'Test Accuracy: {accuracy:.2f}'
    print(result)
    mlflow.log_text(result, 'test_accuracy.text')

    return model


def train_lgb(args, 
              train_X: np.ndarray, 
              train_Y: np.ndarray, 
              valid_X: np.ndarray, 
              valid_Y: np.ndarray, 
              test_X: np.ndarray, 
              test_Y: np.ndarray):
    # Create LightGBM datasets
    train_data = lgb.Dataset(train_X, label=train_Y)
    valid_data = lgb.Dataset(valid_X, label=valid_Y)

    # Define LightGBM parameters
    params = {
        'objective': 'multiclass',    # 'binary' for binary classification, 'multiclass' for multi-class
        'metric': 'multi_logloss',  # 'binary_logloss' for binary classification, 'multi_logloss' for multi-class
        'num_class': args.num_class,
        'boosting_type': 'gbdt',
        'learning_rate': args.lr,
        'num_leaves': args.num_leaves,
        'max_depth': args.max_depth,
        'subsample': args.subsample,
        'subsample_freq': args.subsample_freq,
        'colsample_bytree': args.colsample_bytree,
        'is_unbalance': True,
        'seed': args.seed,
        'early_stopping_rounds': args.early_stop_rounds, 
        'verbosity': 1,
    }
    mlflow.log_params(params)

    # Train the LightGBM model
    num_rounds = args.episode  # Adjust as needed
    model = lgb.train(params, 
                      train_data, 
                      num_rounds, 
                      valid_sets=[train_data, valid_data])

    mlflow.lightgbm.log_model(model, 'model')

    # Make predictions on the test set
    y_pred = model.predict(test_X, num_iteration=model.best_iteration)

    y_pred_class = np.argmax(y_pred, axis=1)

    # Evaluate the model
    accuracy = accuracy_score(test_Y, y_pred_class)
    result = f'Test Accuracy: {accuracy:.2f}'
    print(result)
    mlflow.log_text(result, 'test_accuracy.txt')


class CustomDataset(Dataset):
    def __init__(self, train_X: np.ndarray, train_Y: np.ndarray):
        self.x = train_X
        self.y = train_Y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

def train_nn(args, 
             train_X: np.ndarray, 
             train_Y: np.ndarray, 
             valid_X: np.ndarray, 
             valid_Y: np.ndarray, 
             test_X: np.ndarray, 
             test_Y: np.ndarray):

    model = MLP(train_X.shape[1], 25)

    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise ValueError

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    train_dataset = CustomDataset(train_X.astype(np.float32), train_Y)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True)

    valid_dataset = CustomDataset(valid_X.astype(np.float32), valid_Y)
    valid_dataloader = DataLoader(dataset=valid_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False)

    test_dataset = CustomDataset(test_X.astype(np.float32), test_Y)
    test_dataloader = DataLoader(dataset=test_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False)


    print(f'run id: {run.info.run_id}')
    params = {
        'SEED': args.seed,
        'lr': args.lr,
        'episode': args.episode,
        'optimizer': args.optimizer,
        'batch_size': args.batch_size,
        'valid_freq': args.valid_freq,
    }
    mlflow.log_params(params)

    max_acc = 0
    for epoch in tqdm(range(args.episode)):
        model.train()
        train_acc = 0
        epoch_loss = 0.
        for data in train_dataloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device).to(torch.long)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            train_acc += (logits.argmax(dim=-1) == labels).float().sum().item()

        train_acc /= train_X.shape[0]

        log_data = {
            'epoch_loss': epoch_loss,
            'train_acc': train_acc,
        }   

        if epoch % args.valid_freq == 0:
            model.eval()
            valid_near_acc = 0
            valid_acc = 0
            valid_loss = 0
            for data in valid_dataloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device).to(torch.long)

                with torch.no_grad():
                    logits = model(inputs)
                valid_acc += (logits.argmax(dim=-1) == labels).float().sum().item()
                valid_near_acc += near_acc(logits.cpu().numpy(), labels.cpu().numpy())
                valid_loss += loss.item()

            valid_acc /= valid_X.shape[0]
            valid_near_acc /= valid_X.shape[0]

            if valid_acc > max_acc:
                mlflow.pytorch.log_model(model, 'model')

            log_data['valid_acc'] = valid_acc
            log_data['valid_near_acc'] = valid_near_acc

        mlflow.log_metrics(log_data, epoch)

    model_uri = f"runs:/{run.info.run_id}/model"
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()
    test_acc = 0
    test_near_acc = 0
    for data in test_dataloader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device).to(torch.long)

        with torch.no_grad():
            logits = model(inputs)
        test_acc += (logits.argmax(dim=-1) == labels).float().sum().item()
        test_near_acc += near_acc(logits.cpu().numpy(), labels.cpu().numpy())

    test_acc /= test_X.shape[0]
    test_near_acc /= test_X.shape[0]

    result = f'Test accuracy: {test_acc:.5f}\nTest near accuracy: {test_near_acc:.5f}'
    print(result)
    mlflow.log_text(result, 'test_accuracy.txt')


def near_acc(logits: np.ndarray, label: np.ndarray, top_n_rank: int=1):
    acc = 0
    pred = np.argsort(logits)[:, ::-1][:, :top_n_rank]
    for i, num in enumerate(label):
        near_label = [num - 6, num - 5, num - 4, 
                      num - 1, num    , num + 1,
                      num + 4, num + 5, num + 6]
        if len(set(pred[i]) & set(near_label)) > 0:
            acc += 1
    return acc


def gen_behavior_policy_prob(args, 
                             dataset_path: str,
                             valid_X: np.ndarray, 
                             test_X: np.ndarray):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    valid_dataset = CustomDataset(valid_X.astype(np.float32), valid_Y)
    valid_dataloader = DataLoader(dataset=valid_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  drop_last=False)

    test_dataset = CustomDataset(test_X.astype(np.float32), test_Y)
    test_dataloader = DataLoader(dataset=test_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  drop_last=False)

    model_uri = f"runs:/{args.gen_run_id}/model"
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()

    valid_probs = []
    for data in valid_dataloader:
        inputs, _ = data
        inputs = inputs.to(device)

        with torch.no_grad():
            logits = model(inputs)
        prob = F.softmax(logits, dim=1).cpu().numpy()
        valid_probs.append(prob)
    valid_probs = np.vstack(valid_probs)
    np.save(os.path.join(dataset_path, 'behavior_policy_prob_valid.npy'), valid_probs)

    test_probs = []
    for data in test_dataloader:
        inputs, _ = data
        inputs = inputs.to(device)

        with torch.no_grad():
            logits = model(inputs)
        prob = F.softmax(logits, dim=1).cpu().numpy()
        test_probs.append(prob)
    test_probs = np.vstack(test_probs)
    np.save(os.path.join(dataset_path, 'behavior_policy_prob_test.npy'), test_probs)


if __name__ == '__main__':
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    log_path = os.path.join(args.log_dir, args.model, args.dataset_version)
    os.makedirs(args.log_dir, exist_ok=True)

    # Load Dataset
    '''
    train / valid / test dataset are original unnomalized dataset, with action and reward
    train / valid / test data contain (s, a, r, s_, done, SOFA, is_alive) transitions, with normalization
    '''
    dataset_path = f"../data/final_dataset/{args.dataset_version}/reward_type={args.reward_type}"

    # train
    with open(os.path.join(dataset_path, 'train.pkl'), 'rb') as file:
        train_dict = pickle.load(file)
    train_data = train_dict['data']
    train_X = train_data['s']
    train_Y = train_data['a'].astype(np.int32).flatten()


    # validation
    with open(os.path.join(dataset_path, 'valid.pkl'), 'rb') as file:
        valid_dict = pickle.load(file)
    valid_data = valid_dict['data']
    valid_X = valid_data['s']
    valid_Y = valid_data['a'].astype(np.int32).flatten()

    # test
    with open(os.path.join(dataset_path, 'test.pkl'), 'rb') as file:
        test_dict = pickle.load(file)
    test_data = test_dict['data']
    test_X = test_data['s']
    test_Y = test_data['a'].astype(np.int32).flatten()

    mlflow.set_tracking_uri("http://127.0.0.1:8787")
    experiment = mlflow.set_experiment(f"behavior_policy_classification")

    if args.gen_run_id is not None:
        gen_behavior_policy_prob(args, dataset_path, valid_X, test_X)
    else:
        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=f"{args.model}-{args.dataset_version}") as run:
            if args.model == 'xgb':
                model = train_xgb(args, train_X, train_Y, valid_X, valid_Y, test_X, test_Y)
            elif args.model == 'lgb':
                model = train_lgb(args, train_X, train_Y, valid_X, valid_Y, test_X, test_Y)
            elif args.model == 'nn':
                model = train_nn(args, train_X, train_Y, valid_X, valid_Y, test_X, test_Y)
            else:
                raise ValueError
