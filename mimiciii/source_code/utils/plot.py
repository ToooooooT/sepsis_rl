import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from scipy.stats import sem

matplotlib.use('Agg')  # Set the backend to Agg

import os
import numpy as np


def plot_action_distribution(action_selections, log_dir):
    '''
    Args:
        action_selections: the frequency of each action be selected
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.hist(range(25), weights=action_selections, bins=np.arange(26)-0.5)

    ax.set_xlabel('action index')
    ax.set_ylabel('freq')
    ax.set_xticks(range(0, 25))

    ax.set_title(f'action distribution')

    plt.savefig(os.path.join(log_dir, f'valid_action_distribution.png'))
    plt.close()


def animation_action_distribution(hists, log_dir):
    '''
    Args:
        hists: a list of each validation of the frequency of each action be selected
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)

    def update(i):
        ax.clear()
        ax.hist(range(25), weights=hists[i], bins=np.arange(26)-0.5)
        ax.set_xlabel('action index')
        ax.set_ylabel('freq')
        ax.set_xticks(range(0, 25))
        ax.set_title(f'action distribution {i}')

    ani = FuncAnimation(fig, update, frames=len(hists), interval=200)
    ani.save(os.path.join(log_dir, 'valid action distribution.gif'), writer='imagemagick')
    plt.close()


def plot_estimate_value(expert_val, policy_val, log_dir, freq):
    '''
    Args:
        expert_val: list of estimate return value of expert policy
        policy_val: list of estimate return value of learned policy
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(list(range(len(policy_val))), policy_val)
    ax.plot(list(range(len(expert_val))), expert_val)
    ax.legend(['policy', 'expert'],loc='best')

    if freq > 1:
        ax.set_xlabel(f'epoch * {freq}')
    else:
        ax.set_xlabel(f'epoch')
    ax.set_ylabel('expected return')

    ax.set_title('policy vs expert return')

    plt.savefig(os.path.join(log_dir, 'valid estimate value.png'))
    plt.close()


def plot_action_dist(actions, test_data_unnorm, log_dir):
    '''
    actions                 : policy action (tensor)
    test_data_unnorm        : original expert dataset unnormalize (DataFrame)
    '''
    actions_low = [0] * 25
    actions_mid = [0] * 25
    actions_high = [0] * 25
    actions_all = [0] * 25
    for index in test_data_unnorm.index:
        # action distribtuion
        if test_data_unnorm.loc[index, 'SOFA'] <= 5:
            actions_low[int(actions[index])] += 1
        elif test_data_unnorm.loc[index, 'SOFA'] < 15:
            actions_mid[int(actions[index])] += 1
        else:
            actions_high[int(actions[index])] += 1
        actions_all[int(actions[index])] += 1

    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16,4))
    ax1.hist(range(25), weights=actions_low, bins=np.arange(26)-0.5)
    ax1.set_xticks(range(0, 25))
    ax1.tick_params(axis='x', labelsize=6)
    ax1.set_title('low SOFA')

    ax2.hist(range(25), weights=actions_mid, bins=np.arange(26)-0.5)
    ax2.set_xticks(range(0, 25))
    ax2.tick_params(axis='x', labelsize=6)
    ax2.set_title('mid SOFA')

    ax3.hist(range(25), weights=actions_high, bins=np.arange(26)-0.5)
    ax3.set_xticks(range(0, 25))
    ax3.tick_params(axis='x', labelsize=6)
    ax3.set_title('high SOFA')

    ax4.hist(range(25), weights=actions_all, bins=np.arange(26)-0.5)
    ax4.set_xticks(range(0, 25))
    ax4.tick_params(axis='x', labelsize=6)
    ax4.set_title('all')

    plt.savefig(os.path.join(log_dir, 'test_action_distribution.png'))
    plt.close()


def plot_pos_neg_action_dist(positive_traj, negative_traj, log_dir):
    # negative_traj['policy action'].hist(bins=np.arange(26)-0.5)
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))
    weight = [0] * 25
    tmp = negative_traj['policy action'].value_counts()
    for i in tmp.index:
        weight[i] = tmp[i]
    ax1.hist(range(25), weights=weight, bins=np.arange(26)-0.5)
    ax1.set_xticks(range(0, 25))
    ax1.tick_params(axis='x', labelsize=6)
    ax1.set_title('negative trajectories policy action')

    weight = [0] * 25
    tmp = negative_traj['action'].value_counts()
    for i in tmp.index:
        weight[i] = tmp[i]
    ax2.hist(range(25), weights=weight, bins=np.arange(26)-0.5)
    ax2.set_xticks(range(0, 25))
    ax2.tick_params(axis='x', labelsize=6)
    ax2.set_title('negative trajectories expert action')

    weight = [0] * 25
    tmp = positive_traj['policy action'].value_counts()
    for i in tmp.index:
        weight[i] = tmp[i]
    ax3.hist(range(25), weights=weight, bins=np.arange(26)-0.5)
    ax3.set_xticks(range(0, 25))
    ax3.tick_params(axis='x', labelsize=6)
    ax3.set_title('positive trajectories policy action')

    weight = [0] * 25
    tmp = positive_traj['action'].value_counts()
    for i in tmp.index:
        weight[i] = tmp[i]
    ax4.hist(range(25), weights=weight, bins=np.arange(26)-0.5)
    ax4.set_xticks(range(0, 25))
    ax4.tick_params(axis='x', labelsize=6)
    ax4.set_title('positive trajectories expert action')
    plt.savefig(os.path.join(log_dir, 'pos_neg_action_compare.png'))
    plt.close()


def plot_diff_action_SOFA_dist(positive_traj, negative_traj, log_dir):
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12,12))

    tmp = positive_traj[positive_traj['action'] != positive_traj['policy action']]['SOFA'].value_counts()
    weight = [0] * 25
    for i in tmp.index:
        i = int(i)
        weight[i] = tmp[i]

    ax1.hist(range(25), weights=weight, bins=np.arange(26)-0.5)
    ax1.set_xticks(range(0, 25))
    ax1.tick_params(axis='x', labelsize=6)
    ax1.set_title('positive trajectories different action SOFA distribution')

    tmp = positive_traj[positive_traj['action'] == positive_traj['policy action']]['SOFA'].value_counts()
    weight = [0] * 25
    for i in tmp.index:
        i = int(i)
        weight[i] = tmp[i]

    ax2.hist(range(25), weights=weight, bins=np.arange(26)-0.5)
    ax2.set_xticks(range(0, 25))
    ax2.tick_params(axis='x', labelsize=6)
    ax2.set_title('positive trajectories same action SOFA distribution')

    tmp = negative_traj[negative_traj['action'] != negative_traj['policy action']]['SOFA'].value_counts()
    weight = [0] * 25
    for i in tmp.index:
        i = int(i)
        weight[i] = tmp[i]

    ax3.hist(range(25), weights=weight, bins=np.arange(26)-0.5)
    ax3.set_xticks(range(0, 25))
    ax3.tick_params(axis='x', labelsize=6)
    ax3.set_title('negative trajectories different action SOFA distribution')

    tmp = negative_traj[negative_traj['action'] == negative_traj['policy action']]['SOFA'].value_counts()
    weight = [0] * 25
    for i in tmp.index:
        i = int(i)
        weight[i] = tmp[i]

    ax4.hist(range(25), weights=weight, bins=np.arange(26)-0.5)
    ax4.set_xticks(range(0, 25))
    ax4.tick_params(axis='x', labelsize=6)
    ax4.set_title('negative trajectories same action SOFA distribution')


    plt.savefig(os.path.join(log_dir, 'diff_action_SOFA_dist.png'))
    plt.close()


def plot_diff_action(positive_traj, negative_traj, log_dir):
    f, ax = plt.subplots(5, 5, figsize=(32,32))

    for i in range(5):
        for j in range(5):
            weight = [0] * 25
            idx = i * 5 + j
            tmp = positive_traj[positive_traj['action'] == idx]['policy action'].value_counts()
            for k in tmp.index:
                weight[int(k)] = tmp[int(k)]

            ax[i][j].hist(range(25), weights=weight, bins=np.arange(26)-0.5)
            ax[i][j].set_xticks(range(0, 25))
            ax[i][j].tick_params(axis='x', labelsize=6)
            ax[i][j].set_title(f'expert action: {i * 5 + j}')

    plt.savefig(os.path.join(log_dir, 'pos_diff_action_compare.png'))
    plt.close()

    f, ax = plt.subplots(5, 5, figsize=(32,32))

    for i in range(5):
        for j in range(5):
            weight = [0] * 25
            idx = i * 5 + j
            tmp = negative_traj[negative_traj['action'] == idx]['policy action'].value_counts()
            for k in tmp.index:
                weight[int(k)] = tmp[int(k)]

            ax[i][j].hist(range(25), weights=weight, bins=np.arange(26)-0.5)
            ax[i][j].set_xticks(range(0, 25))
            ax[i][j].tick_params(axis='x', labelsize=6)
            ax[i][j].set_title(f'expert action: {i * 5 + j}')

    plt.savefig(os.path.join(log_dir, 'neg_diff_action_compare.png'))
    plt.close()


def sliding_mean(data_array, window=1):
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


def plot_survival_rate(expected_return, id_index_map, test_data_unnorm, log_dir):
    '''
    reference: https://github.com/CaryLi666/ID3QNE-algorithm/blob/main/experiment/survival%20rate/main_shengcunlv-37.py
    '''
    survive = np.zeros((len(id_index_map),))
    for i, id in enumerate(id_index_map.keys()):
        index = id_index_map[id][0]
        survive[i] = 1.0 if test_data_unnorm.loc[index, 'died_in_hosp'] != 1.0 and \
                    test_data_unnorm.loc[index, 'mortality_90d'] != 1.0 and \
                    test_data_unnorm.loc[index, 'died_within_48h_of_out_time'] != 1.0 else 0

    bin_medians = []
    mort = []
    mort_std = []
    i = -7
    while i <= 25:
        count = survive[np.logical_and(expected_return >= i - 0.5, expected_return <= i + 0.5)]
        try:
            res = sum(count) / len(count)
            if len(count) >= 2:
                bin_medians.append(i)
                mort.append(res)
                mort_std.append(sem(count))
        except ZeroDivisionError:
            pass
        i += 1

    plt.plot(bin_medians, sliding_mean(mort), color='g')
    plt.fill_between(bin_medians, sliding_mean(mort) - 1 * mort_std,
                     sliding_mean(mort) + 1 * mort_std, color='palegreen')

    x_r = [i / 1.0 for i in range(-7, 27, 3)]
    y_r = [i / 10.0 for i in range(0, 11, 1)]
    plt.xticks(x_r)
    plt.yticks(y_r)

    plt.title('Survival Rate v.s. Expected Return')
    plt.xlabel("Expected Return")
    plt.ylabel("Survival Rate")
    plt.savefig(os.path.join(log_dir, 'survival_rate.png'))
    plt.close()


def plot_expected_return_distribution(expected_return, log_dir):
    expected_return = np.round(expected_return).astype(np.int32)
    # igonre the outlier expected return
    expected_return = np.clip(expected_return, -25, 25)
    max_return = expected_return.max()
    min_return = expected_return.min()
    expected_return_count = np.zeros((max_return - min_return + 1))
    unique_vals, counts = np.unique(expected_return, return_counts=True)
    expected_return_count[unique_vals - min_return] = counts

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111)

    ax.hist(range(min_return, max_return + 1), weights=expected_return_count, bins=np.arange(min_return, max_return + 1)-0.5)

    ax.set_xlabel('expected return')
    ax.set_ylabel('count')
    ax.set_xticks(range(min_return, max_return + 1))

    ax.set_title(f'expected return distribution')

    plt.savefig(os.path.join(log_dir, f'expected_return_distribution.png'))
    plt.close()
