import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from scipy.stats import sem

matplotlib.use('Agg')  # Set the backend to Agg

import os
import numpy as np
import pandas as pd


def animation_action_distribution(hists, log_dir):
    '''
    Args:
        hists: a list of each validation of the frequency of each action be selected
    '''
    f, ax = plt.subplots(1, 1)

    def update(i):
        ax.clear()
        ax.bar(range(25), height=hists[i])
        ax.set_xlabel('action index')
        ax.set_ylabel('freq')
        ax.set_xticks(range(0, 25))
        ax.set_title(f'action distribution {i}')

    ani = FuncAnimation(f, update, frames=len(hists), interval=200)
    ani.save(os.path.join(log_dir, 'valid action distribution.gif'), writer='imagemagick')
    plt.close()


def plot_estimate_value(policy_val: np.ndarray, names: list, log_dir, freq):
    '''
    Args:
        policy_val: estimate return value of learned policy during training process; expected shape (k, T)
    '''
    colors = ['blue', 'red']
    f, ax = plt.subplots(1, 1)
    x = np.arange(policy_val.shape[1])
    for i in range(policy_val.shape[0]):
        ax.plot(x, policy_val[i], color=colors[i], label=names[i])
    ax.legend(names, loc='best')

    if freq > 1:
        ax.set_xlabel(f'epoch * {freq}')
    else:
        ax.set_xlabel(f'epoch')
    ax.set_ylabel('expected return')

    ax.set_title('Learned policy expected return')
    plt.savefig(os.path.join(log_dir, 'valid estimate value.png'))
    plt.close()


def plot_action_dist(actions: np.ndarray, dataset: pd.DataFrame, log_dir):
    '''
    Args:
        actions : policy action; expected shape (B, 1)
        dataset : original expert dataset unnormalize (DataFrame)
    '''
    mask_low = dataset['SOFA'] <= 5
    mask_mid = (dataset['SOFA'] > 5) & (dataset['SOFA'] < 15)
    mask_high = dataset['SOFA'] >= 15

    # Count the occurrences of each unique action for each category
    actions_all = np.bincount(actions.ravel().astype(int), minlength=25)
    actions_low = np.bincount(actions[mask_low].ravel().astype(int), minlength=25)
    actions_mid = np.bincount(actions[mask_mid].ravel().astype(int), minlength=25)
    actions_high = np.bincount(actions[mask_high].ravel().astype(int), minlength=25)

    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16,4))
    ax1.bar(range(25), height=actions_low)
    ax1.set_xticks(range(0, 25))
    ax1.tick_params(axis='x', labelsize=6)
    ax1.set_title('low SOFA')

    ax2.bar(range(25), height=actions_mid)
    ax2.set_xticks(range(0, 25))
    ax2.tick_params(axis='x', labelsize=6)
    ax2.set_title('mid SOFA')

    ax3.bar(range(25), height=actions_high)
    ax3.set_xticks(range(0, 25))
    ax3.tick_params(axis='x', labelsize=6)
    ax3.set_title('high SOFA')

    ax4.bar(range(25), height=actions_all)
    ax4.set_xticks(range(0, 25))
    ax4.tick_params(axis='x', labelsize=6)
    ax4.set_title('all')

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'test_action_distribution.png'))
    plt.close()


def plot_pos_neg_action_dist(positive_traj: pd.DataFrame, negative_traj: pd.DataFrame, log_dir):
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))
    height = np.bincount(negative_traj['policy action'], minlength=25)[:25]
    ax1.bar(range(25), height=height)
    ax1.set_xticks(range(0, 25))
    ax1.tick_params(axis='x', labelsize=6)
    ax1.set_title('negative trajectories policy action')

    height = np.bincount(negative_traj['action'], minlength=25)[:25]
    ax2.bar(range(25), height=height)
    ax2.set_xticks(range(0, 25))
    ax2.tick_params(axis='x', labelsize=6)
    ax2.set_title('negative trajectories expert action')

    height = np.bincount(positive_traj['policy action'], minlength=25)[:25]
    ax3.bar(range(25), height=height)
    ax3.set_xticks(range(0, 25))
    ax3.tick_params(axis='x', labelsize=6)
    ax3.set_title('positive trajectories policy action')

    height = np.bincount(positive_traj['action'], minlength=25)[:25]
    ax4.bar(range(25), height=height)
    ax4.set_xticks(range(0, 25))
    ax4.tick_params(axis='x', labelsize=6)
    ax4.set_title('positive trajectories expert action')

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'pos_neg_action_compare.png'))
    plt.close()


def plot_diff_action_SOFA_dist(positive_traj: pd.DataFrame, negative_traj: pd.DataFrame, log_dir):
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12,12))

    height = np.bincount(positive_traj[positive_traj['action'] != positive_traj['policy action']]['SOFA'], minlength=25)[:25]
    ax1.bar(range(25), height=height)
    ax1.set_xticks(range(0, 25))
    ax1.tick_params(axis='x', labelsize=6)
    ax1.set_title('positive trajectories different action SOFA distribution')

    height = np.bincount(positive_traj[positive_traj['action'] == positive_traj['policy action']]['SOFA'], minlength=25)[:25]
    ax2.bar(range(25), height=height)
    ax2.set_xticks(range(0, 25))
    ax2.tick_params(axis='x', labelsize=6)
    ax2.set_title('positive trajectories same action SOFA distribution')

    height = np.bincount(negative_traj[negative_traj['action'] != negative_traj['policy action']]['SOFA'], minlength=25)[:25]
    ax3.bar(range(25), height=height)
    ax3.set_xticks(range(0, 25))
    ax3.tick_params(axis='x', labelsize=6)
    ax3.set_title('negative trajectories different action SOFA distribution')

    height = np.bincount(negative_traj[negative_traj['action'] == negative_traj['policy action']]['SOFA'], minlength=25)[:25]
    ax4.bar(range(25), height=height)
    ax4.set_xticks(range(0, 25))
    ax4.tick_params(axis='x', labelsize=6)
    ax4.set_title('negative trajectories same action SOFA distribution')

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'diff_action_SOFA_dist.png'))
    plt.close()


def plot_diff_action(positive_traj: pd.DataFrame, negative_traj: pd.DataFrame, log_dir):
    f, ax = plt.subplots(5, 5, figsize=(32,32))

    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            height = np.bincount(positive_traj[positive_traj['action'] == idx]['policy action'], minlength=25)[:25]
            ax[i][j].bar(range(25), height=height)
            ax[i][j].set_xticks(range(0, 25))
            ax[i][j].tick_params(axis='x', labelsize=6)
            ax[i][j].set_title(f'expert action: {i * 5 + j}')

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'pos_diff_action_compare.png'))
    plt.close()

    f, ax = plt.subplots(5, 5, figsize=(32,32))

    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            height = np.bincount(negative_traj[negative_traj['action'] == idx]['policy action'], minlength=25)[:25]
            ax[i][j].bar(range(25), height=height)
            ax[i][j].set_xticks(range(0, 25))
            ax[i][j].tick_params(axis='x', labelsize=6)
            ax[i][j].set_title(f'expert action: {i * 5 + j}')

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'neg_diff_action_compare.png'))
    plt.close()


def sliding_mean(data_array: list, window=1):
    new_list = []
    for i in range(len(data_array)):
        indices = range(max(i - window, 0),
                        min(i + window, len(data_array)))
        avg = 0
        for j in indices:
            avg += data_array[j]
        avg /= float(len(indices))
        new_list.append(avg)
    return np.array(new_list)


def plot_survival_rate(expected_return: np.ndarray, id_index_map: dict, dataset: pd.DataFrame, name: list, log_dir):
    '''
    Args:
        expected_return : expected shape (k, N); k is number of estimators
        id_index_map    : indexes of each icustayid (dict)
        dataset         : original dataset
        name            : list of name of estimator

    reference: https://github.com/CaryLi666/ID3QNE-algorithm/blob/main/experiment/survival%20rate/main_shengcunlv-37.py
    '''
    n = len(name)
    if expected_return.ndim == 1:
        expected_return = expected_return.reshape(1, -1)
    assert(n == expected_return.shape[0])

    survive = np.zeros((len(id_index_map),))
    for i, id in enumerate(id_index_map.keys()):
        index = id_index_map[id][0]
        survive[i] = 1.0 if dataset.loc[index, 'died_in_hosp'] != 1.0 and \
                    dataset.loc[index, 'mortality_90d'] != 1.0 and \
                    dataset.loc[index, 'died_within_48h_of_out_time'] != 1.0 else 0

    bin_medians = [[] for _ in range(n)]
    mort = [[] for _ in range(n)]
    mort_std = [[] for _ in range(n)]
    for k in range(n):
        i = -25
        while i <= 25:
            count = survive[np.logical_and(expected_return[k] >= i - 0.5, expected_return[k] <= i + 0.5)]
            try:
                res = sum(count) / len(count)
                if len(count) >= 2:
                    bin_medians[k].append(i)
                    mort[k].append(res)
                    mort_std[k].append(sem(count))
            except ZeroDivisionError:
                pass
            i += 0.5

    f, ax = plt.subplots(n, 1, figsize=(16, n * 8))

    if n == 1:
        ax.plot(bin_medians[0], sliding_mean(mort[0]), color='g')
        ax.fill_between(bin_medians[0], sliding_mean(mort[0]) - 1 * mort_std[0],
                        sliding_mean(mort[0]) + 1 * mort_std[0], color='palegreen')

        x_r = [i / 1.0 for i in range(-25, 27, 3)]
        y_r = [i / 10.0 for i in range(0, 11, 1)]
        ax.set_xticks(x_r)
        ax.set_yticks(y_r)
        ax.set_title(f'{name[0]} Survival Rate v.s. Expected Return')
        ax.set_xlabel("Expected Return")
        ax.set_ylabel("Survival Rate")
    else:
        for k in range(n):
            ax[k].plot(bin_medians[k], sliding_mean(mort[k]), color='g')
            ax[k].fill_between(bin_medians[k], sliding_mean(mort[k]) - 1 * mort_std[k],
                            sliding_mean(mort[k]) + 1 * mort_std[k], color='palegreen')

            x_r = [i / 1.0 for i in range(-25, 27, 3)]
            y_r = [i / 10.0 for i in range(0, 11, 1)]
            ax[k].set_xticks(x_r)
            ax[k].set_yticks(y_r)
            ax[k].set_title(f'{name[k]} Survival Rate v.s. Expected Return')
            ax[k].set_xlabel("Expected Return")
            ax[k].set_ylabel("Survival Rate")

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'survival_rate.png'))
    plt.close()


def plot_expected_return_distribution(expected_return: np.ndarray, name: list, log_dir):
    '''
    Args:
        expected_return : expected shape (k, N); k is number of estimators
        name            : list of name of estimator
    '''
    clip_val = 25
    n = len(name)
    if expected_return.ndim == 1:
        expected_return = expected_return.reshape(1, -1)
    assert(n == expected_return.shape[0])
    expected_return = np.round(expected_return).astype(np.int32)
    # igonre the outlier expected return
    expected_return = np.clip(expected_return, -clip_val, clip_val)
    expected_return_count = np.apply_along_axis(
        lambda x: np.bincount(x + clip_val, minlength=2 * clip_val + 1),
        axis=1,
        arr=expected_return
    )

    f, ax = plt.subplots(n, 1, figsize=(16, n * 8))

    if n > 1:
        for i in range(n):
            ax[i].bar(range(-clip_val, clip_val + 1), height=expected_return_count[i])
            ax[i].set_xlabel('expected return')
            ax[i].set_ylabel('count')
            ax[i].set_xticks(range(-clip_val, clip_val + 1))
            ax[i].set_title(f'{name[i]} expected return distribution')
    else:
        ax.bar(range(-clip_val, clip_val + 1), height=expected_return_count[0])
        ax.set_xlabel('expected return')
        ax.set_ylabel('count')
        ax.set_xticks(range(-clip_val, clip_val + 1))
        ax.set_title(f'{name[0]} expected return distribution')

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f'expected_return_distribution.png'))
    plt.close()


def make_df_diff(test_dataset: pd.DataFrame, vaso_vals, iv_vals):
    iv_diff = test_dataset['input_4hourly'].values - test_dataset['policy iv'].replace({i: iv_vals[i] for i in range(5)}).values
    vaso_diff = test_dataset['max_dose_vaso'].values - test_dataset['policy vaso'].replace({i: vaso_vals[i] for i in range(5)}).values
    df_diff = pd.DataFrame()
    df_diff['mort'] = test_dataset['died_in_hosp']
    df_diff['iv_diff'] = iv_diff
    df_diff['vaso_diff'] = vaso_diff
    return df_diff


def make_iv_plot_data(df_diff: pd.DataFrame):
    bin_medians_iv = []
    mort_iv = []
    mort_std_iv= []
    i = -800
    while i <= 900:
        count = df_diff.loc[(df_diff['iv_diff'] > i - 50) & (df_diff['iv_diff'] < i + 50)]
        try:
            res = sum(count['mort']) / float(len(count))
            if len(count) >= 2:
                bin_medians_iv.append(i)
                mort_iv.append(res)
                mort_std_iv.append(sem(count['mort']))
        except ZeroDivisionError:
            pass
        i += 100
    return bin_medians_iv, mort_iv, mort_std_iv


def make_vaso_plot_data(df_diff):
    bin_medians_vaso = []
    mort_vaso= []
    mort_std_vaso= []
    i = -0.6
    while i <= 0.6:
        count = df_diff.loc[(df_diff['vaso_diff'] > i - 0.05) & (df_diff['vaso_diff'] < i + 0.05)]
        try:
            res = sum(count['mort'])/float(len(count))
            if len(count) >= 2:
                bin_medians_vaso.append(i)
                mort_vaso.append(res)
                mort_std_vaso.append(sem(count['mort']))
        except ZeroDivisionError:
            pass
        i += 0.1
    return bin_medians_vaso, mort_vaso, mort_std_vaso


def plot_action_diff_survival_rate(train_dataset: pd.DataFrame, test_dataset: pd.DataFrame, log_dir):
    vaso_vals = [0]
    vaso_vals.extend(train_dataset['max_dose_vaso'][train_dataset['max_dose_vaso'] > 0].quantile([0.125, 0.375, 0.625, 0.875]))
    iv_vals = [0]
    iv_vals.extend(train_dataset['input_4hourly'][train_dataset['input_4hourly'] > 0].quantile([0.125, 0.375, 0.625, 0.875]))
    # get low, mid, high SOFA dataset
    df_test_low = test_dataset[test_dataset['SOFA'] <= 5]
    df_test_mid = test_dataset[(test_dataset['SOFA'] > 5) & (test_dataset['SOFA'] < 15)]
    df_test_high = test_dataset[test_dataset['SOFA'] >= 15]
    # get low, mid, high action diff
    df_diff_low = make_df_diff(df_test_low, vaso_vals, iv_vals)
    df_diff_mid = make_df_diff(df_test_mid, vaso_vals, iv_vals)
    df_diff_high = make_df_diff(df_test_high, vaso_vals, iv_vals)
    bin_med_iv_low, mort_iv_low, mort_std_iv_low = make_iv_plot_data(df_diff_low)
    bin_med_vaso_low, mort_vaso_low, mort_std_vaso_low = make_vaso_plot_data(df_diff_low)
    bin_med_iv_mid, mort_iv_mid, mort_std_iv_mid = make_iv_plot_data(df_diff_mid)
    bin_med_vaso_mid, mort_vaso_mid, mort_std_vaso_mid = make_vaso_plot_data(df_diff_mid)
    bin_med_iv_high, mort_iv_high, mort_std_iv_high = make_iv_plot_data(df_diff_high)
    bin_med_vaso_high, mort_vaso_high, mort_std_vaso_high = make_vaso_plot_data(df_diff_high)

    def diff_plot(med_vaso, mort_vaso, std_vaso, med_iv, mort_iv, std_iv, col, title, log_dir):
        f, ((ax1, ax2)) = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(10,4))
        step = 2
        if col == 'r':
            fillcol = 'lightsalmon'
        elif col == 'g':
            fillcol = 'palegreen'
            step = 1
        elif col == 'b':
            fillcol = 'lightblue'
        ax1.plot(med_vaso, sliding_mean(mort_vaso), color=col)
        ax1.fill_between(med_vaso, sliding_mean(mort_vaso) - 1 * std_vaso,  
                        sliding_mean(mort_vaso) + 1 * std_vaso, color=fillcol)
        ax1.set_title(title + ': Vasopressors')
        x_r = [i / 10.0 for i in range(-6, 8, 2)]
        y_r = [i / 20.0 for i in range(0, 20, step)]
        ax1.set_xticks(x_r)
        ax1.set_yticks(y_r)
        ax1.grid()

        ax2.plot(med_iv, sliding_mean(mort_iv), color=col)
        ax2.fill_between(med_iv, sliding_mean(mort_iv) - 1 * std_iv,  
                        sliding_mean(mort_iv) + 1 * std_iv, color=fillcol)
        ax2.set_title(title + ': IV fluids')
        x_iv = [i for i in range(-800, 900, 400)]
        ax2.set_xticks(x_iv)
        ax2.grid()

        f.text(0.3, -0.03, 'Difference between optimal and physician vasopressor dose', ha='center', fontsize=10)
        f.text(0.725, -0.03, 'Difference between optimal and physician IV dose', ha='center', fontsize=10)
        f.text(0.05, 0.5, 'Observed Mortality', va='center', rotation='vertical', fontsize = 10)

        plt.savefig(os.path.join(log_dir, f'diff_action_mortality_{title}.png'))
        plt.close()
        
    diff_plot(bin_med_vaso_low, mort_vaso_low, mort_std_vaso_low, 
          bin_med_iv_low, mort_iv_low, mort_std_iv_low, 'b', 'Low SOFA', log_dir)
    diff_plot(bin_med_vaso_mid, mort_vaso_mid, mort_std_vaso_mid, 
          bin_med_iv_mid, mort_iv_mid, mort_std_iv_mid, 'g', 'Medium SOFA', log_dir)
    diff_plot(bin_med_vaso_high, mort_vaso_high, mort_std_vaso_high, 
          bin_med_iv_high, mort_iv_high, mort_std_iv_high, 'r', 'High SOFA', log_dir)
