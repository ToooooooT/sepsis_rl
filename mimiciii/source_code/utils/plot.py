import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import os

import numpy as np

def plot_training_loss(tds, log_dir):
    '''
    Args:
        tds: list of loss
    '''
    fig, ax = plt.subplots()

    ax.plot(tds)

    ax.set_xlabel('epoch * 1000')
    ax.set_ylabel('loss')

    ax.set_title('training loss')

    plt.savefig(os.path.join(log_dir, 'training_loss.png'))


def plot_action_distribution(action_selections, log_dir):
    '''
    Args:
        action_selections: the frequency of each action be selected
    '''
    fig, ax = plt.subplots()

    ax.hist(range(25), weights=action_selections, bins=np.arange(26)-0.5)

    ax.set_xlabel('action index')
    ax.set_ylabel('freq')
    ax.set_xticks(range(0, 25))

    ax.set_title(f'action distribution')

    plt.savefig(os.path.join(log_dir, f'valid_action_distribution.png'))


def animation_action_distribution(hists, log_dir):
    '''
    Args:
        hists: a list of each validation of the frequency of each action be selected
    '''
    fig, ax = plt.subplots()

    def update(i):
        ax.clear()
        ax.hist(range(25), weights=hists[i], bins=np.arange(26)-0.5)
        ax.set_xlabel('action index')
        ax.set_ylabel('freq')
        ax.set_xticks(range(0, 25))
        ax.set_title(f'action distribution {i}')

    ani = FuncAnimation(fig, update, frames=len(hists), interval=200)
    ani.save(os.path.join(log_dir, 'valid action distribution.gif'), writer='imagemagick')


def plot_estimate_value(expert_val, policy_val, log_dir):
    '''
    Args:
        expert_val: list of estimate return value of expert policy
        policy_val: list of estimate return value of learned policy
    '''
    fig, ax = plt.subplots()

    ax.plot(list(range(len(policy_val))), policy_val)
    ax.plot(list(range(len(expert_val))), expert_val)
    ax.legend(['policy', 'expert'],loc='best')

    ax.set_xlabel('epoch * 1000')
    ax.set_ylabel('estimate value')

    ax.set_title('policy vs expert value')

    plt.savefig(os.path.join(log_dir, 'valid estimate value.png'))


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