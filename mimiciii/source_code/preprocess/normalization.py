######################################################################################
# Import package
######################################################################################

import pandas as pd
import numpy as np
import random
import os
from tqdm import tqdm
from math import tanh
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
from argparse import ArgumentParser

matplotlib.use('Agg')  # Set the backend to Agg

pd.options.mode.chained_assignment = None

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset_version", type=str, help="dataset version", default='v0_20937')
    parser.add_argument("--icustayid", type=str, help="a pickle file of dictionary contain icustayid list correspond to train, valid, test", default=None)
    parser.add_argument("--beta", type=float, help="reward function coefficient for type 1", default=0.6)
    parser.add_argument("--min_max", action='store_true', help="normalize with min, max aggregation dataset")
    args = parser.parse_args()
    return args

def get_reward(s, s_, reward_type):
    if reward_type == 0:
        if s_ is None or s['icustayid'] != s_['icustayid']:
            r = -15 if int(s['mortality_90d']) else 15
        else:
            c0, c1, c2 = -0.025, -0.125, -2
            s_sofa = s['SOFA']
            s_lactate = s['Arterial_lactate']
            s_sofa_ = s_['SOFA']
            s_lactate_ = s_['Arterial_lactate']

            r = c0 * int(s_sofa_ == s_sofa and s_sofa_ > 0) + \
                    c1 * (s_sofa_ - s_sofa) + c2 * tanh(s_lactate_ - s_lactate)
    elif reward_type == 1:
        if s_ is None or s['icustayid'] != s_['icustayid']:
            r = -24 if int(s['mortality_90d']) else 24
        else:
            s_sofa = s['SOFA']
            s_sofa_ = s_['SOFA']
            r = args.beta * (s_sofa - s_sofa_)
    elif reward_type == 2:
        if s_ is None or s['icustayid'] != s_['icustayid']:
            r = -24 if int(s['mortality_90d']) else 24
        else:
            c2 = -2
            s_sofa = s['SOFA']
            s_lactate = s['Arterial_lactate']
            s_sofa_ = s_['SOFA']
            s_lactate_ = s_['Arterial_lactate']
            r = args.beta * (s_sofa - s_sofa_) + c2 * tanh(s_lactate_ - s_lactate)
            pass
    return r


def map_action(s, hour):
    iv = s[f'input_{hour}hourly']
    vaso = s[f'max_dose_vaso']
    a = 0
    if iv > 529.757:
        a += 20
    elif iv >= 180.435:
        a += 15
    elif iv >= 50:
        a += 10
    elif iv > 0:
        a += 5

    if vaso > 0.45:
        a += 4
    elif vaso >= 0.225:
        a += 3
    elif vaso >= 0.08:
        a += 2
    elif vaso > 0:
        a += 1
    
    return a


def add_reward(dataset: pd.DataFrame, reward_type):
    dataset['reward'] = [0] * dataset.shape[0]
    for index in tqdm(dataset.index[:-1]):
        s = dataset.loc[index, :]
        s_ = dataset.loc[index + 1, :]
        if s['icustayid'] != s_['icustayid']:
            s_ = None 
        r = get_reward(s, s_, reward_type)
        dataset.loc[index, 'reward'] = r

    # handle last state
    s = dataset.loc[dataset.index[-1], :]
    r = get_reward(s, None, reward_type)
    dataset.loc[dataset.index[-1], 'reward'] = r

def add_action(dataset: pd.DataFrame, hour):
    dataset['action'] = [0] * dataset.shape[0]
    for index in tqdm(dataset.index[:-1]):
        s = dataset.loc[index, :]
        s_ = dataset.loc[index + 1, :]
        if s['icustayid'] != s_['icustayid']:
            s_ = None 
        a = map_action(s, hour)
        dataset.loc[index, 'action'] = a

    # handle last state
    s = dataset.loc[dataset.index[-1], :]
    a = map_action(s, hour)
    dataset.loc[dataset.index[-1], 'action'] = a

def plot_reward(dataset: pd.DataFrame, source_path, name):
    rewards = list()
    for index in dataset.index:
        s = dataset.loc[index, :]
        rewards.append(s['reward'])

    data = pd.DataFrame({'r': rewards})

    fig, ax = plt.subplots()

    # plot data reward distribution
    ax.hist(data['r'], bins=np.arange(53)-0.5)
    ax.set_xticks(range(-26, 26, 2))
    plt.savefig(os.path.join(source_path, f'{name} reward distribution.png'))
    plt.close()

def plot_action(dataset: pd.DataFrame, source_path, name):
    actions, sofa = list(), list()
    for index in dataset.index:
        s = dataset.loc[index, :]
        sofa.append(s['SOFA'])
        actions.append(s['action'])

    data = pd.DataFrame({'a':actions, 'SOFA':sofa})

    # plot data action distribution
    actions_low = data[data['SOFA'] <= 5]['a']
    actions_mid = data[(data['SOFA'] > 5) & (data['SOFA'] < 15)]['a']
    actions_high = data[data['SOFA'] >= 15]['a']
    actions = data['a']

    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16,4))
    ax1.hist(actions_low, bins=np.arange(26)-0.5)
    ax1.set_xticks(range(0, 25))
    ax1.tick_params(axis='x', labelsize=6)
    ax1.set_title('low SOFA')

    ax2.hist(actions_mid, bins=np.arange(26)-0.5)
    ax2.set_xticks(range(0, 25))
    ax2.tick_params(axis='x', labelsize=6)
    ax2.set_title('mid SOFA')

    ax3.hist(actions_high, bins=np.arange(26)-0.5)
    ax3.set_xticks(range(0, 25))
    ax3.tick_params(axis='x', labelsize=6)
    ax3.set_title('high SOFA')

    ax4.hist(actions, bins=np.arange(26)-0.5)
    ax4.set_xticks(range(0, 25))
    ax4.tick_params(axis='x', labelsize=6)
    ax4.set_title('all')

    plt.savefig(os.path.join(source_path, f'{name} action distribution.png'))
    plt.close()


def save_info(dataset: pd.DataFrame, source_path: str, name: str):
    expected_return = dataset.groupby('icustayid').sum()['reward'].mean()
    print(f'{name} expected return: {expected_return}')

    mortaliy_rate = dataset.groupby('icustayid').mean()['mortality_90d'].mean()
    print(f'{name} mortality rate: {mortaliy_rate}')

    with open(os.path.join(source_path, f'{name}_info.txt'), 'w') as f:
        f.write(f'expected return: {expected_return}\n')
        f.write(f'mortality rate: {mortaliy_rate}')


def normalization(period, dataset: pd.DataFrame, is_min_max: bool):
    feats = ['PaO2_FiO2']
    if is_min_max:
        feats = ['PaO2_FiO2_min', 'PaO2_FiO2_max']
    for feat in feats:
        dataset[feat].replace(np.inf, np.nan, inplace=True)
        dataset[feat].replace(-np.inf, np.nan, inplace=True)
        mean_PaO2_FiO2 = dataset[feat].mean(skipna=True)
        dataset[feat].fillna(value=mean_PaO2_FiO2, inplace=True)

    binary_features = ['gender', 're_admission', 'mechvent']
    # norm_features = ['age', 'elixhauser', 'GCS', 'SOFA', 'Albumin', 'Arterial_pH', 
    #                     'Calcium', 'Glucose', 'Hb', 'Magnesium', 'PTT', 'Potassium', 
    #                     'Arterial_BE', 'HCO3', 'Arterial_lactate', 'CO2_mEqL', 'Ionised_Ca', 'PT', 
    #                     'Platelets_count', 'WBC_count', 'Chloride', 'DiaBP', 'SysBP', 'MeanBP', 
    #                     'paCO2', 'paO2', 'FiO2_1', 'RR', 'Temp_C', 'Weight_kg', 'HR', 'Shock_Index', 
    #                     'SIRS', 'PaO2_FiO2', 'cumulated_balance']
    # log_norm_features = ['SGPT', 'BUN', 'INR', 'Creatinine', 'SGOT', 'Total_bili', 'SpO2', 'input_total',
    #                     f'input_{period}hourly', 'output_total', f'output_{period}hourly', 'bloc']

    norm_features = ['GCS', 'Albumin', 'Arterial_pH', 
                        'Calcium', 'Glucose', 'Hb', 'Magnesium', 'PTT', 'Potassium', 
                        'Arterial_BE', 'HCO3', 'CO2_mEqL', 'Ionised_Ca', 'PT', 
                        'Platelets_count', 'WBC_count', 'Chloride', 'DiaBP', 'SysBP', 'MeanBP', 
                        'paCO2', 'paO2', 'FiO2_1', 'RR', 'Temp_C', 'Weight_kg', 'HR', 'Shock_Index', 
                        'SIRS', 'PaO2_FiO2'] 
    if is_min_max:
        tmp = []
        for feature in norm_features:
            tmp.append(feature + '_min')
            tmp.append(feature + '_max')
        norm_features = tmp

    norm_features += ['age', 'elixhauser', 'cumulated_balance', 'SOFA', 'Arterial_lactate']

    log_norm_features = ['SGPT', 'BUN', 'INR', 'Creatinine', 'SGOT', 'Total_bili', 'SpO2']
    if is_min_max:
        tmp = []
        for feature in log_norm_features:
            tmp.append(feature + '_min')
            tmp.append(feature + '_max')
        log_norm_features = tmp

    log_norm_features += ['input_total', f'input_{period}hourly', 'output_total', f'output_{period}hourly', 'bloc']

    dataset[binary_features] = dataset[binary_features] - 0.5
        
    for feature in norm_features:
        avg = dataset[feature].mean()
        std = dataset[feature].std()
        dataset[feature] = (dataset[feature] - avg) / std 

    dataset[log_norm_features] = np.log(0.1 + dataset[log_norm_features])
    for feature in log_norm_features:
        avg = dataset[feature].mean()
        std = dataset[feature].std()
        dataset[feature] = (dataset[feature] - avg) / std


def process_dataset(dataset: pd.DataFrame, unnorm_dataset: pd.DataFrame, save_path):
    drop_column = ['charttime', 'median_dose_vaso', 'input_total', 'icustayid', 'died_in_hosp', 'mortality_90d',
                'died_within_48h_of_out_time', 'delay_end_of_record_and_discharge_or_death',
                'input_4hourly', 'max_dose_vaso', 'reward', 'action', 'SOFA_CV']
    data = {'s': [], 'a': [], 'r': [], 's_': [], 'a_': [], 'bloc_num': [], 
            'done': [], 'SOFA': [], 'SOFA_CV': [], 'is_alive': []}
    state_dim = len(set(dataset.columns) - set(drop_column))
    id_index_map = defaultdict(list)
    terminal_index = set()
    bloc_num = 1
    for index in tqdm(dataset.index[:-1]):
        s = dataset.iloc[index, :]
        s_ = dataset.loc[index + 1, :]
        a = s['action']
        a_ = s_['action']
        r = s['reward']
        SOFA = unnorm_dataset.loc[index, 'SOFA']
        SOFA_CV = unnorm_dataset.loc[index, 'SOFA_CV']
        if s['icustayid'] != s_['icustayid']:
            done = 1
            s_ = [0] * state_dim
            a_ = 0 # useless action
            terminal_index.add(index)
            bloc_num += 1
        else:
            done = 0
            s_ = s_.drop(drop_column)
        id_index_map[s['icustayid']].append(index)
        s.drop(drop_column, inplace=True)
        data['s'].append(s)
        data['a'].append(a)
        data['r'].append(r)
        data['s_'].append(s_)
        data['a_'].append(a_)
        data['done'].append(done)
        data['SOFA'].append(SOFA)
        data['SOFA_CV'].append(SOFA_CV)
        data['is_alive'].append(1 if r != -15 else 0)
        data['bloc_num'].append(bloc_num)

    index = dataset.index[-1]
    terminal_index.add(index)
    s = dataset.loc[index, :]
    s_ = [0] * state_dim
    a = s['action']
    a_ = 0 # useless action
    r = s['reward']
    SOFA = unnorm_dataset.loc[index, 'SOFA']
    SOFA_CV = unnorm_dataset.loc[index, 'SOFA_CV']
    id_index_map[s['icustayid']].append(index)
    s.drop(drop_column, inplace=True)
    done = 1
    data['s'].append(s)
    data['a'].append(a)
    data['r'].append(r)
    data['s_'].append(s_)
    data['a_'].append(a_)
    data['done'].append(done)
    data['SOFA'].append(SOFA)
    data['SOFA_CV'].append(SOFA_CV)
    data['is_alive'].append(1 if r != -15 else 0)
    data['bloc_num'].append(bloc_num)

    data['s'] = np.array(data['s'])
    data['a'] = np.array(data['a']).reshape(-1, 1)
    data['r'] = np.array(data['r']).reshape(-1, 1)
    data['s_'] = np.array(data['s_'])
    data['a_'] = np.array(data['a_']).reshape(-1, 1)
    data['done'] = np.array(data['done']).reshape(-1, 1)
    data['SOFA'] = np.array(data['SOFA']).reshape(-1, 1)
    data['SOFA_CV'] = np.array(data['SOFA_CV']).reshape(-1, 1)
    data['is_alive'] = np.array(data['is_alive']).reshape(-1, 1)
    data['bloc_num'] = np.array(data['bloc_num']).reshape(-1, 1)
    data['iv'] = data['a'] // 5
    data['vaso'] = data['a'] % 5

    save_obj = {'data': data, 'id_index_map': id_index_map, 'terminal_index': terminal_index}
    with open(save_path, 'wb') as file:
        pickle.dump(save_obj, file)


if __name__ == '__main__':
    args = parse_args()

    source_path = f'../../data/final_dataset/{args.dataset_version}'

    hour = 4

    dataset = pd.read_csv(os.path.join(source_path, f'dataset_{hour}.csv'))

    add_action(dataset, hour)

    ################################################################
    # split dataset 8:1:1 (train:valid:test)
    ################################################################
    if args.icustayid is None:
        icustayid = list(dataset['icustayid'].unique())
        random.shuffle(icustayid)
        test_id = set(icustayid[:int(len(icustayid) / 10)])
        valid_id = set(icustayid[int(len(icustayid) / 10):int(len(icustayid) / 5)])
        train_id = set(icustayid[int(len(icustayid) / 5):])
    else:
        with open(args.icustayid, 'rb') as file:
            icustayid = pickle.load(file)
        train_id = set(icustayid['train'])
        valid_id = set(icustayid['valid'])
        test_id = set(icustayid['test'])

        # check all train_id, valid_id, test_id is in the dataset
        icustayid_set = set(dataset['icustayid'].unique())
        assert train_id.issubset(icustayid_set)
        assert valid_id.issubset(icustayid_set)
        assert test_id.issubset(icustayid_set)
        # check train, valid, test are not overlap with each other
        assert train_id.isdisjoint(valid_id)
        assert valid_id.isdisjoint(test_id)
        assert train_id.isdisjoint(test_id)
        # replace original dataset
        dataset = dataset[[x in train_id or x in valid_id or x in test_id for x in dataset['icustayid']]]
        dataset.to_csv(os.path.join(source_path, f'dataset_{hour}.csv'))

    with open(os.path.join(source_path, f'dataset_info.txt'), 'w') as f:
        t = dataset.groupby('icustayid').mean()['mortality_90d']
        f.write(f'dead: {(t == 1).sum()}\n')
        f.write(f'alive: {(t == 0).sum()}\n')
        f.write(f'total: {t.shape[0]}\n')
        f.write(f'mortality_rate: {t.mean() * 100:.3f}%')

    train_dataset_origin = dataset[[x in train_id for x in dataset['icustayid']]]
    valid_dataset_origin = dataset[[x in valid_id for x in dataset['icustayid']]]
    test_dataset_origin = dataset[[x in test_id for x in dataset['icustayid']]]

    train_dataset_origin = train_dataset_origin.reset_index(drop=True)
    valid_dataset_origin = valid_dataset_origin.reset_index(drop=True)
    test_dataset_origin = test_dataset_origin.reset_index(drop=True)

    plot_action(train_dataset_origin, source_path, 'train')
    plot_action(valid_dataset_origin, source_path, 'valid')
    plot_action(test_dataset_origin, source_path, 'test')

    for reward_type in range(3):
        reward_path = os.path.join(source_path, f'reward_type={reward_type}')
        os.makedirs(reward_path, exist_ok=True)

        train_dataset = train_dataset_origin.copy(deep=True)
        valid_dataset = valid_dataset_origin.copy(deep=True)
        test_dataset = test_dataset_origin.copy(deep=True)

        add_reward(train_dataset, reward_type)
        add_reward(valid_dataset, reward_type)
        add_reward(test_dataset, reward_type)

        unnorm_train_dataset = train_dataset.copy(deep=True).reset_index(drop=True)
        unnorm_valid_dataset = valid_dataset.copy(deep=True).reset_index(drop=True)
        unnorm_test_dataset = test_dataset.copy(deep=True).reset_index(drop=True)

        unnorm_train_dataset.to_csv(os.path.join(reward_path, f'train.csv'), index=False)
        unnorm_valid_dataset.to_csv(os.path.join(reward_path, f'valid.csv'), index=False)
        unnorm_test_dataset.to_csv(os.path.join(reward_path, f'test.csv'), index=False)

        normalization(hour, train_dataset, args.min_max)
        normalization(hour, valid_dataset, args.min_max)
        normalization(hour, test_dataset, args.min_max)

        plot_reward(train_dataset, reward_path, 'train')
        plot_reward(valid_dataset, reward_path, 'valid')
        plot_reward(test_dataset, reward_path, 'test')

        save_info(train_dataset, reward_path, 'train')
        save_info(valid_dataset, reward_path, 'valid')
        save_info(test_dataset, reward_path, 'test')

        process_dataset(train_dataset, unnorm_train_dataset, os.path.join(reward_path, f'train.pkl'))
        process_dataset(valid_dataset, unnorm_valid_dataset, os.path.join(reward_path, f'valid.pkl'))
        process_dataset(test_dataset, unnorm_test_dataset, os.path.join(reward_path, f'test.pkl'))
