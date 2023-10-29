import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import os

from agents import BaseAgent
from utils import ExperienceReplayMemory, PrioritizedReplayMemory

class SAC(BaseAgent):
    def __init__(self, static_policy=False, env=None, config=None, log_dir='./logs') -> None:
        super().__init__(config=config, env=env, log_dir=log_dir)
        self.device = config.device

        # algorithm control
        self.priority_replay = config.USE_PRIORITY_REPLAY

        # misc agent variables
        self.gamma = config.GAMMA
        self.lr = config.LR

        self.is_gradient_clip = config.IS_GRADIENT_CLIP

        # memory
        self.target_net_update_freq = config.TARGET_NET_UPDATE_FREQ
        self.experience_replay_size = config.EXP_REPLAY_SIZE
        self.batch_size = config.BATCH_SIZE
        self.priority_alpha = config.PRIORITY_ALPHA
        self.priority_beta_start = config.PRIORITY_BETA_START
        self.priority_beta_frames = config.PRIORITY_BETA_FRAMES

        # update target network parameter
        self.tau = config.TAU

        # environment
        self.num_feats = env['num_feats']
        self.num_actions = env['num_actions']

        self.declare_memory()

        # network
        self.static_policy = static_policy

        self.declare_networks()
        self.target_qf1.load_state_dict(self.qf1.state_dict())
        self.target_qf2.load_state_dict(self.qf2.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.lr)
        self.target_qf1.eval()
        self.target_qf2.eval()

        # move model to device
        self.actor = self.actor.to(self.device)
        self.qf1 = self.qf1.to(self.device)
        self.qf2 = self.qf2.to(self.device)
        self.target_qf1 = self.target_qf1.to(self.device)
        self.target_qf2 = self.target_qf2.to(self.device)

        # Entropy regularization coefficient
        self.autotune = config.AUTOTUNE
        if self.autotune:
            self.target_entropy = config.TARGET_ENTROPY_SCALE * \
                torch.log(1 / torch.tensor(np.array(self.num_actions), dtype=torch.float, device=self.device))
            self.log_alpha = torch.zeros(1, dtype=torch.float, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=self.lr, eps=1e-4)
        else:
            self.alpha = config.ALPHA

        self.behav_clone = config.BEHAVIOR_CLONING

        if static_policy:
            self.actor.eval()
            self.qf1.eval()
            self.qf2.eval()
        else:
            self.actor.train()
            self.qf1.train()
            self.qf2.train()


    def declare_networks(self):
        self.actor: nn.Module = None
        self.qf1: nn.Module = None
        self.qf2: nn.Module = None
        self.target_qf1: nn.Module = None
        self.target_qf2: nn.Module = None
        raise NotImplementedError # override thid function


    def declare_memory(self):
        self.memory = ExperienceReplayMemory(self.experience_replay_size) if not self.priority_replay else PrioritizedReplayMemory(self.experience_replay_size, self.priority_alpha, self.priority_beta_start, self.priority_beta_frames)


    def save(self):
        os.makedirs(self.log_dir, exist_ok=True)
        torch.save({
            'actor': self.actor.state_dict(),
            'qf1': self.qf1.state_dict(),
            'qf2': self.qf2.state_dict(),
            }, 
            os.path.join(self.log_dir, 'model.pth')
        )

    
    def load(self):
        fname = os.path.join(self.log_dir, 'model.pth')

        if os.path.isfile(fname):
            checkpoint = torch.load(fname)
        else:
            raise ValueError

        self.actor.load_state_dict(checkpoint['actor'])


    def append_to_replay(self, s, a, r, s_, done):
        self.memory.push((s, a, r, s_, done))


    def prep_minibatch(self):
        '''
        Returns:
            states: expected shape (B, S)
            actions: expected shape (B, D)
            rewards: expected shape (B, 1)
            next_states: expected shape (B, S)
            dones: expected shape (B, 1)
            indices: a list of index
            weights: expected shape (B,)
        '''
        # random transition batch is taken from replay memory
        transitions, indices, weights = self.memory.sample(self.batch_size)
        
        states, actions, rewards, next_states, dones = zip(*transitions)

        states = torch.tensor(np.array(states), device=self.device, dtype=torch.float)
        actions = torch.tensor(np.array(actions), device=self.device, dtype=torch.int64).view(-1, 1)
        rewards = torch.tensor(np.array(rewards), device=self.device, dtype=torch.float).view(-1, 1)
        next_states = torch.tensor(np.array(next_states), device=self.device, dtype=torch.float)
        dones = torch.tensor(np.array(dones), device=self.device, dtype=torch.float).view(-1, 1)

        # check shape
        assert states.shape == (self.batch_size, self.num_feats)
        assert actions.shape == (self.batch_size, 1)
        assert rewards.shape == (self.batch_size, 1)
        assert next_states.shape == (self.batch_size, self.num_feats)
        assert dones.shape == (self.batch_size, 1)
        
        return states, actions, rewards, next_states, dones, indices, weights

    def compute_critic_loss(self, states, actions, rewards, next_states, dones, indices, weights):
        with torch.no_grad():
            _, _, next_state_log_pi, next_state_action_probs = self.get_action(next_states)
            qf1_next_target = self.target_qf1(next_states)
            qf2_next_target = self.target_qf2(next_states)
            # we can use the action probabilities instead of MC sampling to estimate the expectation
            min_qf_next_target = next_state_action_probs * (
                torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            )
            # adapt Q-target for discrete Q-function
            min_qf_next_target = min_qf_next_target.sum(dim=1, keepdim=True)
            next_q_value = rewards + (1 - dones) * self.gamma * (min_qf_next_target)

        qf1_values = self.qf1(states).gather(1, actions)
        qf2_values = self.qf2(states).gather(1, actions)

        if self.priority_replay:
            # include actor loss or not?
            diff1 = next_q_value - qf1_values
            diff2 = next_q_value - qf2_values
            diff = diff1 + diff2
            self.memory.update_priorities(indices, diff.detach().squeeze().abs().cpu().numpy().tolist())
            qf1_loss = ((0.5 * diff1.pow(2))* weights).mean()
            qf2_loss = ((0.5 * diff2.pow(2))* weights).mean()
        else:
            qf1_loss = F.mse_loss(qf1_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_values, next_q_value)

        qf_loss = qf1_loss + qf2_loss
        return qf_loss

    def compute_actor_loss(self, states):
        _, _, log_pi, action_probs = self.get_action(states)
        with torch.no_grad():
            qf1_values = self.qf1(states)
            qf2_values = self.qf2(states)
            min_qf_values = torch.min(qf1_values, qf2_values)
        # no need for reparameterization, the expectation can be calculated for discrete actions
        actor_loss = (action_probs * (self.alpha * log_pi - min_qf_values)).mean()
        return actor_loss, action_probs, log_pi

    def update(self, t):
        # ref: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_atari.py
        self.actor.train()
        self.qf1.train()
        self.qf2.train()
        states, actions, rewards, next_states, dones, indices, weights = self.prep_minibatch()
        # update critic 
        qf_loss = self.compute_critic_loss(states, actions, rewards, next_states, dones, indices, weights)
        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()
        # update actor 
        actor_loss, action_probs, log_pi = self.compute_actor_loss()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.autotune:
            # Entropy regularization coefficient training
            # reuse action probabilities for temperature loss
            alpha_loss = (action_probs.detach() * (-self.log_alpha * (log_pi + self.target_entropy).detach())).mean()
            self.a_optimizer.zero_grad()
            alpha_loss.backward()
            self.a_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
        # update the target network
        if t % self.target_net_update_freq == 0:
            self.update_target_model(self.target_qf1, self.qf1)
            self.update_target_model(self.target_qf2, self.qf2)

        return {'qf_loss': qf_loss.detach().cpu().item(), 'actor_loss': actor_loss.detach().cpu().item()}


    def get_action(self, x):
        logits = self.actor(x)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return action, logits, log_prob, action_probs


    def update_target_model(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
