# Spring 2023, 535515 Reinforcement Learning
# HW1: REINFORCE and baseline

import os
import gym
from collections import namedtuple
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# Define a useful tuple (optional)
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# Define a tensorboard writer
        
class Policy(nn.Module):
    """
        Implement both policy network and the value network in one model
        - Note that here we let the actor and value networks share the first layer
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
        TODO:
            1. Initialize the network (including the GAE parameters, shared layer(s), the action layer(s), and the value layer(s))
            2. Random weight initialization of each layer
    """
    
    def __init__(self):
        super(Policy, self).__init__()
        
        # Extract the dimensionality of state and action spaces
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.hidden_size = 128
        self.double()

        self.epsilon = 0.5

        self.l1 = nn.Sequential(
            nn.Linear(self.observation_dim, self.hidden_size),
            nn.LeakyReLU()
        )
        self.actor_l2 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU()
        )
        self.value_l2 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU()
        )
        self.actor_l3 = nn.Sequential(
            nn.Linear(self.hidden_size, self.action_dim),
            nn.Softmax()
        )
        self.value_l3 = nn.Linear(self.hidden_size, 1)
        # initialize weight
        nn.init.kaiming_normal_(self.l1[0].weight)
        nn.init.kaiming_normal_(self.actor_l2[0].weight)
        nn.init.kaiming_normal_(self.value_l2[0].weight)
        nn.init.kaiming_normal_(self.actor_l3[0].weight)
        nn.init.kaiming_normal_(self.value_l3.weight)
        
        # action & reward memory
        self.saved_actions = []
        self.rewards = []
        self.gae = GAE(gamma=0.999, lambda_=1, num_steps=None)

    def forward(self, state):
        """
            Forward pass of both policy and value networks
            - The input is the state, and the outputs are the corresponding 
              action probability distirbution and the state value
        """
        
        y = self.l1(state)
        actor_y = self.actor_l2(y)
        action_prob = self.actor_l3(actor_y)
        value_y = self.value_l2(y)
        state_value = self.value_l3(value_y)

        return action_prob, state_value


    def select_action(self, state):
        """
            Select the action given the current state
            - The input is the state, and the output is the action to apply 
            (based on the learned stochastic policy)
            TODO:
                1. Implement the forward pass for both the action and the state value
        """
        
        action_prob, state_value = self.forward(state)
        m = Categorical(action_prob)
        action = m.sample()

        # save to action buffer
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        return action.item()


    def calculate_loss(self, gamma=0.999):
        """
            Calculate the loss (= policy loss + value loss) to perform backprop later
        """
        
        # Initialize the lists and variables
        R = 0
        saved_actions = self.saved_actions[::-1]
        self.rewards.append(0)
        rewards = self.rewards[::-1]
        policy_losses = [] 
        value_losses = [] 
        returns = []
        n = len(saved_actions)

        values = [x.value for x in self.saved_actions]
        gae = self.gae(self.rewards, values, None)
        for i, (log_prob, state_value) in enumerate(saved_actions):
            if i == 0:
                # terminal state
                value_losses.append(F.mse_loss(state_value, torch.tensor([0.0])))
                returns.append(rewards[i])
            else:
                value_losses.append(F.mse_loss(torch.tensor([[rewards[i] + gamma * returns[i - 1]]], dtype=torch.float), state_value))
                returns.append(rewards[i] + gamma * returns[i - 1])
                policy_losses.append(-(gamma**(n - i - 1)) * log_prob * gae[i])

        loss = sum(value_losses) + sum(policy_losses)
        
        return loss

    def clear_memory(self):
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]

class GAE:
    def __init__(self, gamma, lambda_, num_steps):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.num_steps = num_steps          # set num_steps = None to adapt full batch

    def __call__(self, rewards, values, done):
        """
            Implement Generalized Advantage Estimation (GAE) for your value prediction

            values: state_values
        """
        # terminal state
        rewards = rewards[:-1]
        n = len(rewards)
        gae = [-values[-1]]
        for i in range(n - 1, -1, -1):
            td_error = rewards[i] + self.gamma * values[i + 1] - values[i]
            gae.append(td_error + self.gamma * self.lambda_ * gae[-1])

        return gae


def test(name, n_episodes=1000):
    """
        Test the learned model (no change needed)
    """     
    model = Policy()
    
    model.load_state_dict(torch.load(os.path.join('./preTrained', name)))
    
    # render = True
    max_episode_len = 10000
    expert_data = {'s': [], 's_': [], 'a': [], 'r': [], 'done': []}
    for i_episode in range(1, n_episodes+1):
        trajectory = {'s': [], 'a': [], 'r': []}
        state = env.reset()
        running_reward = 0
        for t in range(max_episode_len+1):
            state = torch.tensor(state).view(1, -1)
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            running_reward += reward
            trajectory['s'].append(state)
            trajectory['a'].append(action)
            trajectory['r'].append(reward)
            if done:
                break
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
        expert_data['s'] += trajectory['s']
        expert_data['s_'] += (trajectory['s'][1:] + trajectory['s'][0:1]) # last state is useless, so I use fhe first state
        expert_data['a'] += trajectory['a']
        expert_data['r'] += trajectory['r']
        expert_data['done'] += ([0] * (len(trajectory['s']) - 1) + [1])
    env.close()

    with open('./dataset/expert.pkl', 'wb') as file:
        pickle.dump(expert_data, file)


if __name__ == '__main__':
    # For reproducibility, fix the random seed
    random_seed = 10  
    lr = 0.001
    env = gym.make('LunarLander-v2')
    env.seed(random_seed)  
    torch.manual_seed(random_seed)  
    test('expert.pth', n_episodes=5000)