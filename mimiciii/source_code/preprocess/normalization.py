######################################################################################
# Import package
######################################################################################

import pandas as pd
import numpy as np
import random
import os
from math import tanh
import matplotlib.pyplot as plt

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


def add_reward_action(dataset, hour):
    dataset['reward'] = [0] * dataset.shape[0]
    dataset['action'] = [0] * dataset.shape[0]
    for index in dataset.index[:-1]:
        s = dataset.loc[index, :]
        s_ = dataset.loc[index + 1, :]
        if s['icustayid'] != s_['icustayid']:
            s_ = None
            r = -15 if int(s['died_in_hosp']) or int(s['died_within_48h_of_out_time']) or int(s['mortality_90d']) else 15
        else:
            r = get_reward(s, s_)
        a = map_action(s, hour)
        dataset.loc[index, 'reward'] = r
        dataset.loc[index, 'action'] = a

    # handle last state
    s = dataset.loc[dataset.index[-1], :]
    r = -15 if int(s['died_in_hosp']) or int(s['died_within_48h_of_out_time']) or int(s['mortality_90d']) else 15
    a = map_action(s, hour)
    dataset.loc[dataset.index[-1], 'reward'] = r
    dataset.loc[dataset.index[-1], 'action'] = a

def plot_reward_action(dataset, name):
    rewards, actions, sofa = list(), list(), list()
    for index in dataset.index:
        s = dataset.loc[index, :]
        sofa.append(s['SOFA'])
        rewards.append(s['reward'])
        actions.append(s['action'])

    data = pd.DataFrame({'a':actions, 'SOFA':sofa, 'r':rewards})

    fig, ax = plt.subplots()

    # plot data reward distribution
    ax.hist(data['r'], bins=40)
    ax.set_xticks(range(-15, 16, 2))
    plt.savefig(f'../log/{name} reward distribution.png')

    # plot data action distribution
    actions_low = data[data['SOFA'] <= 5]['a']
    actions_mid = data[data['SOFA'] > 5][data['SOFA'] < 15]['a']
    actions_high = data[data['SOFA'] >= 15]['a']
    actions = data['a']

    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16,4))
    ax1.hist(actions_low, bins=25)
    ax1.set_xticks(range(0, 25))
    ax1.tick_params(axis='x', labelsize=6)
    ax1.set_title('low SOFA')

    ax2.hist(actions_mid, bins=25)
    ax2.set_xticks(range(0, 25))
    ax2.tick_params(axis='x', labelsize=6)
    ax2.set_title('mid SOFA')

    ax3.hist(actions_high, bins=25)
    ax3.set_xticks(range(0, 25))
    ax3.tick_params(axis='x', labelsize=6)
    ax3.set_title('high SOFA')

    ax4.hist(actions, bins=25)
    ax4.set_xticks(range(0, 25))
    ax4.tick_params(axis='x', labelsize=6)
    ax4.set_title('all')

    plt.savefig(f'../log/{name} action distribution.png')


def normalization(period, dataset):
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
                        f'input_{period}hourly', 'output_total', f'output_{period}hourly']

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

    plot_reward_action(train_dataset, 'train')
    plot_reward_action(valid_dataset, 'valid')
    plot_reward_action(test_dataset, 'test')
    normalization(hour, train_dataset)
    normalization(hour, valid_dataset)
    normalization(hour, test_dataset)

    train_dataset.to_csv(os.path.join(source_path, 'train_4.csv'), index=False)
    valid_dataset.to_csv(os.path.join(source_path, 'valid_4.csv'), index=False)
    test_dataset.to_csv(os.path.join(source_path, 'test_4.csv'), index=False)