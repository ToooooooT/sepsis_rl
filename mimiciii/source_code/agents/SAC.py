import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os

from agents import BaseAgent
from utils import ExperienceReplayMemory, PrioritizedReplayMemory

class SAC(BaseAgent):
    def __init__(self, static_policy=False, env=None, config=None, log_dir='./log', agent_dir='./saved_agents') -> None:
        super().__init__(config=config, env=env, log_dir=log_dir, agent_dir=agent_dir)

        # algorithm control
        self.priority_replay = config.USE_PRIORITY_REPLAY

        # misc agent variables
        self.gamma = config.GAMMA
        self.lr = config.LR

        # memory
        self.target_net_update_freq = config.TARGET_NET_UPDATE_FREQ
        self.experience_replay_size = config.EXP_REPLAY_SIZE
        self.batch_size = config.BATCH_SIZE
        self.priority_alpha = config.PRIORITY_ALPHA
        self.priority_beta_start = config.PRIORITY_BETA_START
        self.priority_beta_frames = config.PRIORITY_BETA_FRAMES

        # environment
        self.num_feats = env['num_feats']
        self.num_actions = env['num_actions']

        # loss
        self.reg_lambda = config.REG_LAMBDA
        self.reward_threshold = 20

        self.update_count = 0

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

        # update target network parameter
        self.tau = 0.001

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
        raise NotImplementedError


    def declare_memory(self):
        self.memory = ExperienceReplayMemory(self.experience_replay_size) if not self.priority_replay else PrioritizedReplayMemory(self.experience_replay_size, self.priority_alpha, self.priority_beta_start, self.priority_beta_frames)


    def save(self):
        os.makedirs(self.agent_dir, exist_ok=True)
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


    def append_to_replay(self, s, a, r, s_, SOFA):
        self.memory.push((s, a, r, s_, SOFA))


    def prep_minibatch(self):
        '''
        Returns:
            batch_state: expected shape (B, S)
            batch_action: expected shape (B, D)
            batch_reward: expected shape (B, 1)
            batch_next_state: expected shape (B, S)
            batch_SOFA: expected shape (B, 1)
            batch_done: expected shape (B, 1)
            indices: a list of index
            weights: expected shape (B,)
        '''
        # random transition batch is taken from replay memory
        transitions, indices, weights = self.memory.sample(self.batch_size)
        
        batch_state, batch_action, batch_reward, batch_next_state, batch_SOFA = zip(*transitions)

        batch_state = torch.tensor(np.array(batch_state), device=self.device, dtype=torch.float)
        batch_action = torch.tensor(np.array(batch_action), device=self.device, dtype=torch.int64).view(-1, 1)
        batch_reward = torch.tensor(np.array(batch_reward), device=self.device, dtype=torch.float).view(-1, 1)
        batch_SOFA= torch.tensor(np.array(batch_SOFA), device=self.device, dtype=torch.float).view(-1, 1)
        
        batch_done = torch.tensor(tuple(map(lambda s: 0 if s is not None else 1, batch_next_state)), device=self.device, dtype=torch.float).view(-1, 1)
        batch_next_state = torch.tensor(np.array([s if s is not None else [0] * self.num_feats for s in batch_next_state]), 
                                        device=self.device, dtype=torch.float).view(-1, self.num_feats)

        return batch_state, batch_action, batch_reward, batch_next_state, batch_SOFA, batch_done, indices, weights


    def update(self, t):
        # ref: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_atari.py
        self.actor.train()
        self.qf1.train()
        self.qf2.train()

        states, actions, rewards, next_states, SOFAs, dones, indices, weights = self.prep_minibatch()

        # critic training
        '''
            loss function = E[Q_double-target - Q_estimate]^2 + lambda * max(|Q_estimate| - Q_threshold, 0)
            Q_double-target = reward + gamma * Q_double-target(next_state, argmax_a(Q(next_state, a)))
            Q_threshold = 20
            when die reward -15 else +15
        '''
        with torch.no_grad():
            _, _, next_state_log_pi, next_state_action_probs = self.actor.get_action(next_states)
            qf1_next_target = self.target_qf1(next_states)
            qf2_next_target = self.target_qf2(next_states)
            # we can use the action probabilities instead of MC sampling to estimate the expectation
            min_qf_next_target = next_state_action_probs * (
                torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            )
            # adapt Q-target for discrete Q-function
            min_qf_next_target = min_qf_next_target.sum(dim=1, keepdim=True)
            next_q_value = rewards + (1 - dones) * self.gamma * (min_qf_next_target)

        qf1_values = self.qf1(states)
        qf2_values = self.qf2(states)
        qf1_a_values = qf1_values.gather(1, actions)
        qf2_a_values = qf2_values.gather(1, actions)

        td_error1 = (next_q_value - qf1_a_values).pow(2)
        td_error2 = (next_q_value - qf2_a_values).pow(2)
        td_error = td_error1 + td_error2

        if self.priority_replay:
            self.memory.update_priorities(indices, (td_error / 10).detach().squeeze().abs().cpu().numpy().tolist()) # ?
            qf1_loss = 0.5 * (td_error1 * weights).mean()
            qf2_loss = 0.5 * (td_error2 * weights).mean()
            # qf1_loss = 0.5 * (td_error1 * weights).mean() + self.reg_lambda * max(qf1_a_values.abs().max() - self.reward_threshold, 0)
            # qf2_loss = 0.5 * (td_error2 * weights).mean() + self.reg_lambda * max(qf2_a_values.abs().max() - self.reward_threshold, 0)
        else:
            qf1_loss = 0.5 * td_error1.mean()
            qf2_loss = 0.5 * td_error2.mean()
            # qf1_loss = 0.5 * td_error1.mean() + self.reg_lambda * max(qf1_a_values.abs().max() - self.reward_threshold, 0)
            # qf2_loss = 0.5 * td_error2.mean() + self.reg_lambda * max(qf2_a_values.abs().max() - self.reward_threshold, 0)

        qf_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        # actor training
        _, logits, log_pi, action_probs = self.actor.get_action(states)
        with torch.no_grad():
            qf1_values = self.qf1(states)
            qf2_values = self.qf2(states)
            min_qf_values = torch.min(qf1_values, qf2_values)
        # no need for reparameterization, the expectation can be calculated for discrete actions
        if self.behav_clone:
            clone = (SOFAs < 5).view(-1)
            actor_loss = (action_probs * (self.alpha * log_pi - min_qf_values)).mean() + \
                            F.cross_entropy(action_probs[clone, :], actions.view(-1)[clone])
        else:
            actor_loss = (action_probs * (self.alpha * log_pi - min_qf_values)).mean()

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


    def update_target_model(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
