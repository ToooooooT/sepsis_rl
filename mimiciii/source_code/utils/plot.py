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