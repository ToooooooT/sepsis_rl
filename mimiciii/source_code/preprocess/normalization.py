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

matplotlib.use('Agg')  # Set the backend to Agg

pd.options.mode.chained_assignment = None

def get_reward(s, s_):
    c0, c1, c2 = -0.025, -0.125, -2
    s_sofa = s['SOFA']
    s_lactate = s['Arterial_lactate']
    s_sofa_ = s_['SOFA']
    s_lactate_ = s_['Arterial_lactate']

    r = c0 * int(s_sofa_ == s_sofa and s_sofa_ > 0) + \
            c1 * (s_sofa_ - s_sofa) + c2 * tanh(s_lactate_ - s_lactate)

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


def add_reward_action(dataset: pd.DataFrame, hour):
    dataset['reward'] = [0] * dataset.shape[0]
    dataset['action'] = [0] * dataset.shape[0]
    for index in tqdm(dataset.index[:-1]):
        s = dataset.loc[index, :]
        s_ = dataset.loc[index + 1, :]
        if s['icustayid'] != s_['icustayid']:
            s_ = None
            r = -15 if int(s['mortality_90d']) else 15
        else:
            r = get_reward(s, s_)
        a = map_action(s, hour)
        dataset.loc[index, 'reward'] = r
        dataset.loc[index, 'action'] = a

    # handle last state
    s = dataset.loc[dataset.index[-1], :]
    r = -15 if int(s['mortality_90d']) else 15
    a = map_action(s, hour)
    dataset.loc[dataset.index[-1], 'reward'] = r
    dataset.loc[dataset.index[-1], 'action'] = a

def plot_reward_action(dataset: pd.DataFrame, name):
    rewards, actions, sofa = list(), list(), list()
    for index in dataset.index:
        s = dataset.loc[index, :]
        sofa.append(s['SOFA'])
        rewards.append(s['reward'])
        actions.append(s['action'])

    data = pd.DataFrame({'a':actions, 'SOFA':sofa, 'r':rewards})

    fig, ax = plt.subplots()

    # plot data reward distribution
    ax.hist(data['r'], bins=np.arange(41)-0.5)
    ax.set_xticks(range(-15, 16, 2))
    plt.savefig(f'../log/{name} reward distribution.png')

    # plot data action distribution
    actions_low = data[data['SOFA'] <= 5]['a']
    actions_mid = data[data['SOFA'] > 5][data['SOFA'] < 15]['a']
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

    plt.savefig(f'../log/{name} action distribution.png')


def normalization(period, dataset: pd.DataFrame):
    dataset['PaO2_FiO2'].replace(np.inf, np.nan, inplace=True)
    dataset['PaO2_FiO2'].replace(-np.inf, np.nan, inplace=True)
    mean_PaO2_FiO2 = dataset['PaO2_FiO2'].mean(skipna=True)
    dataset['PaO2_FiO2'].fillna(value=mean_PaO2_FiO2, inplace=True)

    binary_features = ['gender', 're_admission', 'mechvent']
    norm_features = ['age', 'elixhauser', 'GCS', 'SOFA', 'Albumin', 'Arterial_pH', 
                        'Calcium', 'Glucose', 'Hb', 'Magnesium', 'PTT', 'Potassium', 
                        'Arterial_BE', 'HCO3', 'Arterial_lactate', 'CO2_mEqL', 'Ionised_Ca', 'PT', 
                        'Platelets_count', 'WBC_count', 'Chloride', 'DiaBP', 'SysBP', 'MeanBP', 
                        'paCO2', 'paO2', 'FiO2_1', 'RR', 'Temp_C', 'Weight_kg', 'HR', 'Shock_Index', 
                        'SIRS', 'PaO2_FiO2', 'cumulated_balance']
    log_norm_features = ['SGPT', 'BUN', 'INR', 'Creatinine', 'SGOT', 'Total_bili', 'SpO2', 'input_total',
                        f'input_{period}hourly', 'output_total', f'output_{period}hourly', 'bloc']

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
                'input_4hourly', 'max_dose_vaso', 'reward', 'action']
    data = {'s': [], 'a': [], 'r': [], 's_': [], 'a_': [], 'bloc_num': [], 'done': [], 'SOFA': [], 'is_alive': []}
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
        if s['icustayid'] != s_['icustayid']:
            done = 1
            s_ = [0] * state_dim
            a_ = -1
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
        data['is_alive'].append(1 if r != -15 else 0)
        data['bloc_num'].append(bloc_num)

    index = dataset.index[-1]
    terminal_index.add(index)
    s = dataset.loc[index, :]
    s_ = [0] * state_dim
    a = s['action']
    a_ = -1
    r = s['reward']
    SOFA = unnorm_dataset.loc[index, 'SOFA']
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
    data['is_alive'].append(1 if r != -15 else 0)
    data['bloc_num'].append(bloc_num)

    data['s'] = np.array(data['s'])
    data['a'] = np.array(data['a'])
    data['r'] = np.array(data['r'])
    data['s_'] = np.array(data['s_'])
    data['a_'] = np.array(data['a_'])
    data['done'] = np.array(data['done'])
    data['SOFA'] = np.array(data['SOFA'])
    data['is_alive'] = np.array(data['is_alive'])
    data['bloc_num'] = np.array(data['bloc_num'])
    data['iv'] = data['a'] / 5
    data['vaso'] = data['a'] % 5

    save_obj = {'data': data, 'id_index_map': id_index_map, 'terminal_index': terminal_index}
    with open(save_path, 'wb') as file:
        pickle.dump(save_obj, file)


if __name__ == '__main__':
    source_path = '../../data/final_dataset/'

    hour = 4

    dataset = pd.read_csv(os.path.join(source_path, f'dataset_{hour}.csv'))

    add_reward_action(dataset, hour)

    ################################################################
    # split dataset 8:1:1 (train:valid:test)
    ################################################################
    icustayid = list(dataset['icustayid'].unique())
    random.shuffle(icustayid)
    test_id = icustayid[:int(len(icustayid) / 10)]
    valid_id = icustayid[int(len(icustayid) / 10):int(len(icustayid) / 5)]
    train_id = icustayid[int(len(icustayid) / 5):]

    train_dataset = dataset[[x in train_id for x in dataset['icustayid']]]
    valid_dataset = dataset[[x in valid_id for x in dataset['icustayid']]]
    test_dataset = dataset[[x in test_id for x in dataset['icustayid']]]

    train_dataset = train_dataset.reset_index(drop=True)
    valid_dataset = valid_dataset.reset_index(drop=True)
    test_dataset = test_dataset.reset_index(drop=True)

    unnorm_train_dataset = train_dataset.copy(deep=True).reset_index(drop=True)
    unnorm_valid_dataset = valid_dataset.copy(deep=True).reset_index(drop=True)
    unnorm_test_dataset = test_dataset.copy(deep=True).reset_index(drop=True)

    unnorm_train_dataset.to_csv(os.path.join(source_path, 'train_4.csv'), index=False)
    unnorm_valid_dataset.to_csv(os.path.join(source_path, 'valid_4.csv'), index=False)
    unnorm_test_dataset.to_csv(os.path.join(source_path, 'test_4.csv'), index=False)

    plot_reward_action(train_dataset, 'train')
    plot_reward_action(valid_dataset, 'valid')
    plot_reward_action(test_dataset, 'test')
    normalization(hour, train_dataset)
    normalization(hour, valid_dataset)
    normalization(hour, test_dataset)

    process_dataset(train_dataset, unnorm_train_dataset, os.path.join(source_path, 'train.pkl'))
    process_dataset(valid_dataset, unnorm_valid_dataset, os.path.join(source_path, 'valid.pkl'))
    process_dataset(test_dataset, unnorm_test_dataset, os.path.join(source_path, 'test.pkl'))