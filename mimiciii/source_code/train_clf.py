import numpy as np
import pickle
import os
from argparse import ArgumentParser
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--hour", type=int, help="hours of one state", default=4)
    parser.add_argument("--seed", type=int, help="random seed", default=10)
    parser.add_argument("--log_dir", type=str, help="log directory", default='./log/Clf/')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    np.random.seed(args.seed)
    # Load Dataset
    '''
    train / valid / test dataset are original unnomalized dataset, with action and reward
    train / valid / test data contain (s, a, r, s_, done, SOFA, is_alive) transitions, with normalization
    '''
    dataset_path = "../data/final_dataset/"

    # train
    with open(os.path.join(dataset_path, 'train.pkl'), 'rb') as file:
        train_dict = pickle.load(file)
    train_data = train_dict['data']

    # validation
    with open(os.path.join(dataset_path, 'valid.pkl'), 'rb') as file:
        valid_dict = pickle.load(file)
    valid_data = valid_dict['data']

    # test
    with open(os.path.join(dataset_path, 'test.pkl'), 'rb') as file:
        test_dict = pickle.load(file)
    test_data, test_id_index_map = test_dict['data'], test_dict['id_index_map']

    # train logistic regression to predict alive or not
    clf = LogisticRegression(max_iter=1000)
    train_feat = np.concatenate([train_data['s'], train_data['iv'].reshape(-1, 1), train_data['vaso'].reshape(-1, 1)], axis=1)
    clf.fit(train_feat, train_data['is_alive'])
    # validation
    valid_feat = np.concatenate([valid_data['s'], valid_data['iv'].reshape(-1, 1), valid_data['vaso'].reshape(-1, 1)], axis=1)
    y_pred = clf.predict(valid_feat)
    accuracy = accuracy_score(valid_data['is_alive'], y_pred)
    print(f'validation score: {accuracy:.5f}, survival rate: {np.array(y_pred).mean():.5f}')
    # test
    test_feat = np.concatenate([test_data['s'], test_data['iv'].reshape(-1, 1), test_data['vaso'].reshape(-1, 1)], axis=1)
    y_pred = clf.predict(test_feat)
    accuracy = accuracy_score(test_data['is_alive'], y_pred)
    print(f'test score: {accuracy:.5f}, survival rate: {np.array(y_pred).mean():.5f}')
    joblib.dump(clf, os.path.join(args.log_dir, 'LG_clf.sav'))