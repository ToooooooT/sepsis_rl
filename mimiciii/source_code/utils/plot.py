import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from scipy.stats import sem

matplotlib.use('Agg')  # Set the backend to Agg

import os
import numpy as np
import pandas as pd


def plot_action_distribution(action_selections, log_dir):
    '''
    Args:
        action_selections: the frequency of each action be selected
    '''
    plt.bar(range(25), height=action_selections)
    plt.xlabel('action index')
    plt.ylabel('freq')
    plt.xticks(range(0, 25))
    plt.title(f'action distribution')
    plt.savefig(os.path.join(log_dir, f'valid_action_distribution.png'))
    plt.close()


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


def plot_estimate_value(policy_val, log_dir, freq):
    '''
    Args:
        expert_val: list of estimate return value of expert policy
        policy_val: list of estimate return value of learned policy
    '''
    plt.plot(list(range(len(policy_val))), policy_val)
    plt.legend(['policy'], loc='best')

    if freq > 1:
        plt.xlabel(f'epoch * {freq}')
    else:
        plt.xlabel(f'epoch')
    plt.ylabel('expected return')

    plt.title('Learned policy expected return')
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
        i = -7
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

        x_r = [i / 1.0 for i in range(-7, 27, 3)]
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

            x_r = [i / 1.0 for i in range(-7, 27, 3)]
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
