# Spring 2023, 535515 Reinforcement Learning
# HW1: REINFORCE and baseline

import gym
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.optim.lr_scheduler as Scheduler
from torch.utils.tensorboard import SummaryWriter

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
        """
        
        action_prob, state_value = self.forward(state)
        m = Categorical(action_prob)
        action = m.sample()

        # save to action buffer
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        return action.item()


    def get_action(self, state):
        with torch.no_grad():
            action_prob, state_value = self.forward(state)
            m = Categorical(action_prob)
            action = m.sample()

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

def train(lr=0.01):
    """
        Train the model using SGD (via backpropagation)
    """
    
    # Instantiate the policy model and the optimizer
    model = Policy()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    epochs = 10000000
    
    # Learning rate scheduler (optional)
    scheduler = Scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    
    # EWMA reward for tracking the learning progress
    ewma_reward = 0
    max_avg_reward = 0
    # run inifinitely many episodes
    for i_episode in range(epochs):
        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0
        t = 0

        # Uncomment the following line to use learning rate scheduler
        scheduler.step()
        if optimizer.param_groups[0]['lr'] < 1e-5:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-5  # Set it to your desired minimum learning rate
        
        # For each episode, only run 9999 steps to avoid entering infinite loop during the learning process
        
        done = False
        while not done and t < 9999:
            state = torch.tensor(state).view(1, -1)
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            model.rewards.append(reward)
            ep_reward += reward
            t += 1

        # add terminal state
        model.select_action(torch.tensor(state).view(1, -1))
        optimizer.zero_grad()
        loss = model.calculate_loss()
        loss.backward()
        optimizer.step()
        model.clear_memory()
            
        # update EWMA reward and log the results
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        print('Episode {}\tlength: {}\treward: {}\t ewma reward: {}'.format(i_episode, t, ep_reward, ewma_reward))

        #Try to use Tensorboard to record the behavior of your implementation 
        writer.add_scalar('training loss', loss, i_episode)
        writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], i_episode)
        writer.add_scalar('total reward', ep_reward, i_episode)

        if i_episode % 1000 == 0:
            avg_reward = test(model)
            writer.add_scalar('test average reward', avg_reward, i_episode)

            # check if we have "solved" the cart pole problem, use 120 as the threshold in LunarLander-v2
            if avg_reward > max_avg_reward:
                torch.save(model.state_dict(), './preTrained/LunarLander_GAE_{}_best.pth'.format(lr))
                max_avg_reward = avg_reward
            if max_avg_reward > 290:
                print("Solved! Running reward is now {} and "
                        "the last episode runs to {} time steps!".format(ewma_reward, t))
                break

def test(model: Policy, n_episodes=20):
    """
        Test the learned model (no change needed)
    """     
    max_episode_len = 10000
    total_reward = 0
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        running_reward = 0
        for t in range(max_episode_len+1):
            state = torch.tensor(state).view(1, -1)
            action = model.get_action(state)
            state, reward, done, _ = env.step(action)
            running_reward += reward
            if done:
                break
        total_reward += running_reward
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    env.close()
    return total_reward / n_episodes



if __name__ == '__main__':
    # For reproducibility, fix the random seed
    random_seed = 10  
    lr = 0.001
    writer = SummaryWriter(f'./logs/GAE_lr{lr}')
    env = gym.make('LunarLander-v2')
    env.seed(random_seed)  
    torch.manual_seed(random_seed)  
    train(lr)