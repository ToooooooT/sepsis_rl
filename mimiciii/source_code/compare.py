import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_action_dist(positive_traj, negative_traj, folder):
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
    plt.savefig(os.path.join(folder, 'pos_neg_action_compare.png'))
    plt.close()


def plot_diff_action_SOFA_dist(positive_traj, negative_traj, folder):
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


    plt.savefig(os.path.join(folder, 'diff_action_SOFA_dist.png'))
    plt.close()


def plot_diff_action(positive_traj, negative_traj, folder):
    f, ax = plt.subplots(5, 5, figsize=(32,32))

    diff = positive_traj[positive_traj['action'] != positive_traj['policy action']]
    for i in range(5):
        for j in range(5):
            weight = [0] * 25
            idx = i * 5 + j
            tmp = diff[diff['action'] == idx]['policy action'].value_counts()
            for k in tmp.index:
                weight[int(k)] = tmp[int(k)]

            ax[i][j].hist(range(25), weights=weight, bins=np.arange(26)-0.5)
            ax[i][j].set_xticks(range(0, 25))
            ax[i][j].tick_params(axis='x', labelsize=6)
            ax[i][j].set_title(f'expert action: {i * 5 + j}')

    plt.savefig(os.path.join(folder, 'pos_diff_action_compare.png'))
    plt.close()

    f, ax = plt.subplots(5, 5, figsize=(32,32))

    diff = negative_traj[negative_traj['action'] != negative_traj['policy action']]
    for i in range(5):
        for j in range(5):
            weight = [0] * 25
            idx = i * 5 + j
            tmp = diff[diff['action'] == idx]['policy action'].value_counts()
            for k in tmp.index:
                weight[int(k)] = tmp[int(k)]

            ax[i][j].hist(range(25), weights=weight, bins=np.arange(26)-0.5)
            ax[i][j].set_xticks(range(0, 25))
            ax[i][j].tick_params(axis='x', labelsize=6)
            ax[i][j].set_title(f'expert action: {i * 5 + j}')

    plt.savefig(os.path.join(folder, 'neg_diff_action_compare.png'))
    plt.close()


if __name__ == '__main__':
    # train = pd.read_csv('../data/final_dataset/train_4.csv')
    # valid = pd.read_csv('../data/final_dataset/valid_4.csv')
    # test = pd.read_csv('../data/final_dataset/test_4.csv')
    # data = pd.read_csv('../data/final_dataset/dataset_4.csv')
    folder = './log/batch_size-32 episode-150000 use_pri-1 lr-0.0015 reg_lambda-2/'
    predict_test = pd.read_csv(os.path.join(folder, 'test_data_predict.csv'))

    negative_traj = predict_test.query('died_in_hosp == 1.0 | died_within_48h_of_out_time == 1.0 | mortality_90d == 1.0')
    positive_traj = predict_test.query('died_in_hosp != 1.0 & died_within_48h_of_out_time != 1.0 & mortality_90d != 1.0')

    plot_action_dist(positive_traj, negative_traj, folder)
    plot_diff_action_SOFA_dist(positive_traj, negative_traj, folder)
    plot_diff_action(positive_traj, negative_traj, folder)