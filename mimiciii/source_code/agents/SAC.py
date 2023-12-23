import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, kl_divergence
from typing import List, Dict, Tuple
import os

from agents import BaseAgent
from utils import Config
from network import DuellingMLP, PolicyMLP

class SAC(BaseAgent):
    def __init__(self, 
                 env: dict, 
                 config: Config, 
                 log_dir: str='./logs',
                 static_policy: bool=False) -> None:
        super().__init__(config=config, env=env, log_dir=log_dir, static_policy=static_policy)
        
        self.target_qf1.load_state_dict(self.qf1.state_dict())
        self.target_qf2.load_state_dict(self.qf2.state_dict())
        self.target_q_dre.load_state_dict(self.target_q_dre.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.pi_lr)
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + 
                                      list(self.qf2.parameters()) + 
                                      list(self.q_dre.parameters()), lr=self.q_lr)
        self.target_qf1.eval()
        self.target_qf2.eval()
        self.target_q_dre.eval()

        # move model to device
        self.actor = self.actor.to(self.device)
        self.qf1 = self.qf1.to(self.device)
        self.qf2 = self.qf2.to(self.device)
        self.q_dre = self.q_dre.to(self.device)
        self.target_q_dre = self.target_q_dre.to(self.device)
        self.target_qf1 = self.target_qf1.to(self.device)
        self.target_qf2 = self.target_qf2.to(self.device)

        # Entropy regularization coefficient
        self.autotune = config.AUTOTUNE
        if self.autotune:
            self.target_entropy = config.TARGET_ENTROPY_SCALE * \
                torch.log(1 / torch.tensor(np.array(self.num_actions), dtype=torch.float, device=self.device))
            self.log_alpha = torch.zeros(1, dtype=torch.float, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.q_lr, eps=1e-4)
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

    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'actor': self.actor.state_dict(),
            'qf1': self.qf1.state_dict(),
            'qf2': self.qf2.state_dict(),
            'q_dre': self.q_dre.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'q_optimizer': self.q_optimizer.state_dict(),
        }
        if self.autotune:
            checkpoint['log_alpha'] = self.log_alpha.item()
            checkpoint['alpha_optimizer'] = self.alpha_optimizer.state_dict()
        torch.save(checkpoint, os.path.join(self.log_dir, 'checkpoint.pth'))

    def load_checkpoint(self) -> int:
        path = os.path.join(self.log_dir, 'checkpoint.pth')
        if os.path.exists(path):
            checkpoint = torch.load(path)
        else:
            raise FileExistsError
        self.actor.load_state_dict(checkpoint['actor'])
        self.qf1.load_state_dict(checkpoint['qf1'])
        self.qf2.load_state_dict(checkpoint['qf2'])
        self.q_dre.load_state_dict(checkpoint['q_dre'])
        self.target_qf1.load_state_dict(checkpoint['qf1'])
        self.target_qf2.load_state_dict(checkpoint['qf2'])
        self.target_q_dre.load_state_dict(checkpoint['q_dre'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer'])
        if self.autotune:
            self.log_alpha = torch.tensor(np.array([checkpoint['log_alpha']]), dtype=torch.float, requires_grad=True, device=self.device)
            self.alpha_optimizer.load_state_dict = checkpoint['alpha_optimizer']
            self.alpha = self.log_alpha.exp().item()
        return checkpoint['epoch']


    def declare_networks(self):
        self.actor = PolicyMLP(self.num_feats, self.num_actions, hidden_size=self.hidden_size).to(self.device)
        self.qf1 = DuellingMLP(self.num_feats, self.num_actions, hidden_size=self.hidden_size).to(self.device)
        self.qf2 = DuellingMLP(self.num_feats, self.num_actions, hidden_size=self.hidden_size).to(self.device)
        self.q_dre = DuellingMLP(self.num_feats, self.num_actions, hidden_size=self.hidden_size).to(self.device)
        self.target_q_dre = DuellingMLP(self.num_feats, self.num_actions, hidden_size=self.hidden_size).to(self.device)
        self.target_qf1 = DuellingMLP(self.num_feats, self.num_actions, hidden_size=self.hidden_size).to(self.device)
        self.target_qf2 = DuellingMLP(self.num_feats, self.num_actions, hidden_size=self.hidden_size).to(self.device)


    def compute_critic_loss(self, 
                            states: torch.Tensor, 
                            actions: torch.Tensor, 
                            rewards: torch.Tensor, 
                            next_states: torch.Tensor, 
                            dones: torch.Tensor, 
                            indices: List, 
                            weights: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, _, next_state_log_pi, next_state_action_probs = self.get_action_probs(next_states)
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

        # update q function for doubly robust estimator
        q_dre_values = self.q_dre(states).gather(1, actions)
        with torch.no_grad():
            max_next_action = self.q_dre(next_states).max(dim=1)[1].view(-1, 1)
            target_q_dre_values = self.target_q_dre(next_states).gather(1, max_next_action)
        q_dre_loss = F.mse_loss(q_dre_values, rewards + self.gamma * target_q_dre_values * (1 - dones))

        qf_loss = qf1_loss + qf2_loss + q_dre_loss
        return qf_loss

    def compute_actor_loss(self, states) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, _, log_pi, action_probs = self.get_action_probs(states)
        with torch.no_grad():
            qf1_values = self.qf1(states)
            qf2_values = self.qf2(states)
            min_qf_values = torch.min(qf1_values, qf2_values)
        # no need for reparameterization, the expectation can be calculated for discrete actions
        actor_loss = (action_probs * (self.alpha * log_pi - min_qf_values)).mean()
        return actor_loss, action_probs, log_pi

    def update(self, t) -> Dict:
        # ref: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_atari.py
        self.actor.train()
        self.qf1.train()
        self.qf2.train()
        self.q_dre.train()
        states, actions, rewards, next_states, dones, indices, weights = self.prep_minibatch()
        # update critic 
        qf_loss = self.compute_critic_loss(states, actions, rewards, next_states, dones, indices, weights)
        self.q_optimizer.zero_grad()
        qf_loss.backward()
        if self.is_gradient_clip:
            self.gradient_clip_q()
        self.q_optimizer.step()
        # update actor 
        actor_loss, action_probs, log_pi = self.compute_actor_loss(states)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.is_gradient_clip:
            self.gradient_clip_actor()
        self.actor_optimizer.step()

        if self.autotune:
            # Entropy regularization coefficient training
            # reuse action probabilities for temperature loss
            alpha_loss = (action_probs.detach() * (-self.log_alpha.exp() * (log_pi + self.target_entropy).detach())).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
        # update the target network
        if t % self.target_net_update_freq == 0:
            self.update_target_model(self.target_qf1, self.qf1)
            self.update_target_model(self.target_qf2, self.qf2)

        return {'qf_loss': qf_loss.detach().cpu().item(), 
                'actor_loss': actor_loss.detach().cpu().item(), 
                'alpha_loss': alpha_loss.detach().cpu().item()}


    def get_action_probs(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor(states)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return action.view(-1, 1), logits, log_prob, action_probs

    def get_action(self, s, eps=0) -> int:
        '''
        for interacting with environment
        '''
        with torch.no_grad():
            if np.random.random() >= eps or self.static_policy:
                x = torch.tensor(np.array([s]), device=self.device, dtype=torch.float)
                logits = self.actor(x)
                policy_dist = Categorical(logits=logits)
                action = policy_dist.sample()
                return action.item()

        return np.random.randint(0, self.num_actions)

    def gradient_clip_q(self):
        for param in self.qf1.parameters():
            param.grad.data.clamp_(-1, 1) 
        for param in self.qf2.parameters():
            param.grad.data.clamp_(-1, 1) 

    def gradient_clip_actor(self):
        for param in self.actor.parameters():
            param.grad.data.clamp_(-1, 1) 

    def adversarial_state_training(self, 
                                   states: np.ndarray, 
                                   next_states: np.ndarray, 
                                   rewards: np.ndarray,
                                   dones: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''
        for data augmentation
        '''
        # TODO: check this function, currently only augment on states
        states = torch.tensor(states, device=self.device, dtype=torch.float, requires_grad=True)
        next_states = torch.tensor(next_states, device=self.device, dtype=torch.float)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float)

        actions = self.get_action_probs(states)[0]
        next_actions = self.get_action_probs(next_states)[0]

        qf1_values = self.qf1(states).gather(1, actions)
        qf2_values = self.qf2(states).gather(1, actions)
        qf1_next_values = self.qf1(next_states).gather(1, next_actions)
        qf2_next_values = self.qf2(next_states).gather(1, next_actions)
        q_values = torch.min(qf1_values, qf2_values)
        next_q_values = torch.min(qf1_next_values, qf2_next_values)

        loss = (rewards + next_q_values * (1 - dones) - q_values).mean()
        states.grad.zero_()
        loss.backward()
        with torch.no_grad():
            states = states + states.grad * self.adversarial_step
        return states.detach().cpu().numpy(), next_states.cpu().numpy()


class SAC_BC(SAC):
    def __init__(self, 
                 env: dict, 
                 config: Config, 
                 log_dir: str='./logs',
                 static_policy: bool=False) -> None:
        super().__init__(config=config, env=env, log_dir=log_dir, static_policy=static_policy)

        self.actor_lambda = config.ACTOR_LAMBDA
        self.bc_type = config.BC_TYPE
        if self.bc_type == "KL":
            self.bc_kl_beta = config.BC_KL_BETA
            self.log_nu = torch.zeros(1, dtype=torch.float, device=self.device, requires_grad=True)
            self.nu_optimizer = optim.Adam([self.log_nu], lr=self.q_lr, eps=1e-4)


    def compute_actor_loss(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor,
                                                                                       torch.Tensor,
                                                                                       torch.Tensor,
                                                                                       torch.Tensor,
                                                                                       torch.Tensor,
                                                                                       torch.Tensor]:
        _, logits, log_pi, action_probs = self.get_action_probs(states)
        with torch.no_grad():
            qf1_values = self.qf1(states)
            qf2_values = self.qf2(states)
            min_qf_values = torch.min(qf1_values, qf2_values)
        # no need for reparameterization, the expectation can be calculated for discrete actions
        actor_loss = (action_probs * (self.alpha * log_pi - min_qf_values)).mean()
        if self.bc_type == 'cross_entropy':
            bc_loss = F.cross_entropy(action_probs, actions.view(-1))
            kl_loss = None
        else:
            # assume other action probabilities is 0.001 of behavior policy
            behavior_probs = torch.full(action_probs.shape, 0.001, device=self.device)
            behavior_probs.scatter_(1, actions, 1 - 0.001 * (self.num_actions - 1))
            behavior = Categorical(probs=behavior_probs)
            policy = Categorical(logits=logits)
            nu = torch.clamp(self.log_nu.exp(), min=0.0, max=1000000.0)
            # \nu * (\beta - KL(\pi_\phi(a|s) || \pi_{b}(a|s)))
            kl_loss = kl_divergence(policy, behavior).mean()
            bc_loss = nu * (kl_loss - self.bc_kl_beta)
        # TODO: use a suitable coefficient of normalization term
        coef = self.actor_lambda / (action_probs * (self.alpha * log_pi - min_qf_values)).abs().mean().detach()
        total_loss = actor_loss * coef + bc_loss / 6
        return total_loss, actor_loss * coef, bc_loss, kl_loss, action_probs, log_pi

    def update(self, t):
        self.actor.train()
        self.qf1.train()
        self.qf2.train()
        self.q_dre.train()
        states, actions, rewards, next_states, dones, indices, weights = self.prep_minibatch()
        # update critic 
        qf_loss = self.compute_critic_loss(states, actions, rewards, next_states, dones, indices, weights)
        self.q_optimizer.zero_grad()
        qf_loss.backward()
        if self.is_gradient_clip:
            self.gradient_clip_q()
        self.q_optimizer.step()
        # update actor 
        total_loss, actor_loss, bc_loss, kl_loss, action_probs, log_pi = self.compute_actor_loss(states, actions)
        self.actor_optimizer.zero_grad()
        total_loss.backward()
        if self.is_gradient_clip:
            self.gradient_clip_actor()
        self.actor_optimizer.step()

        loss = {'qf_loss': qf_loss.detach().cpu().item(), 
                'actor_loss': actor_loss.detach().cpu().item(), 
                'bc_loss': bc_loss.detach().cpu().item()}

        if self.autotune:
            # Entropy regularization coefficient training
            # reuse action probabilities for temperature loss
            alpha_loss = (action_probs.detach() * (-self.log_alpha.exp() * (log_pi + self.target_entropy).detach())).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
            loss['alpha_loss'] = alpha_loss.detach().cpu().item()
        if self.bc_type == "KL":
            nu = torch.clamp(self.log_nu.exp(), min=0.0, max=1000000.0)
            nu_loss = nu * (kl_loss.detach() - self.bc_kl_beta)
            self.nu_optimizer.zero_grad()
            nu_loss.backward()
            self.nu_optimizer.step()
            loss['nu_loss'] = nu_loss.detach().cpu().item()
            loss['kl_loss'] = kl_loss.detach().cpu().item()
        # update the target network
        if t % self.target_net_update_freq == 0:
            self.update_target_model(self.target_qf1, self.qf1)
            self.update_target_model(self.target_qf2, self.qf2)

        return loss
