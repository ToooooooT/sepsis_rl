import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, kl_divergence
from typing import List, Dict, Tuple
import os

from agents import SAC
from utils import Config

class CQL(SAC):
    def __init__(self, 
                 env: dict, 
                 config: Config, 
                 log_dir: str='./logs',
                 static_policy: bool=False) -> None:
        super().__init__(config=config, env=env, log_dir=log_dir, static_policy=static_policy)

        self.with_lagrange = config.WITH_LAGRANGE
        if self.with_lagrange:
            self.target_action_gap = config.TARGET_ACTION_GAP
            self.log_alpha_prime = torch.zeros(1, dtype=torch.float, device=self.device, requires_grad=True)
            self.alpha_prime_optimizer = optim.Adam([self.log_alpha_prime], lr=self.q_lr, eps=1e-4)

    def save_checkpoint(self, epoch, name: str='checkpoint.pth'):
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
        if self.with_lagrange:
            checkpoint['log_alpha_prime'] = self.log_alpha_prime.item()
            checkpoint['alpha_prime_optimizer'] = self.alpha_prime_optimizer.state_dict()
        torch.save(checkpoint, os.path.join(self.log_dir, name))

    def load_checkpoint(self, name: str='checkpoint.pth') -> int:
        path = os.path.join(self.log_dir, name)
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
        if self.with_lagrange:
            self.log_alpha_prime = torch.tensor(np.array([checkpoint['log_alpha_prime']]), dtype=torch.float, requires_grad=True, device=self.device)
            self.alpha_prime_optimizer.load_state_dict = checkpoint['alpha_prime_optimizer']
        return checkpoint['epoch']

    def compute_critic_loss(self, 
                            states: torch.Tensor, 
                            actions: torch.Tensor, 
                            rewards: torch.Tensor, 
                            next_states: torch.Tensor, 
                            dones: torch.Tensor, 
                            indices: List, 
                            weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qf1_values = self.qf1(states).gather(1, actions)
        qf2_values = self.qf2(states).gather(1, actions)

        # CQL regulariztion loss
        min_qf1_loss_ = torch.logsumexp(self.qf1(states), dim=1).mean() - qf1_values.mean()
        min_qf2_loss_ = torch.logsumexp(self.qf2(states), dim=1).mean() - qf2_values.mean()

        if self.with_lagrange:
            alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0.0, max=1000000.0).detach()
            min_qf1_loss = alpha_prime * (min_qf1_loss_ - self.target_action_gap)
            min_qf2_loss = alpha_prime * (min_qf2_loss_ - self.target_action_gap)

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

        if self.priority_replay:
            # include actor loss, cql regularization loss or not?
            diff1 = next_q_value - qf1_values
            diff2 = next_q_value - qf2_values
            diff = diff1 + diff2
            self.memory.update_priorities(indices, diff.detach().squeeze().abs().cpu().numpy().tolist())
            qf1_loss = ((0.5 * diff1.pow(2))* weights).mean()
            qf2_loss = ((0.5 * diff2.pow(2))* weights).mean()
        else:
            qf1_loss = F.mse_loss(qf1_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_values, next_q_value)

        if self.with_lagrange:
            qf1_loss = qf1_loss + min_qf1_loss
            qf2_loss = qf2_loss + min_qf2_loss
        else:
            qf1_loss = qf1_loss + min_qf1_loss_
            qf2_loss = qf2_loss + min_qf2_loss_

        # update q function for doubly robust estimator
        q_dre_values = self.q_dre(states).gather(1, actions)
        with torch.no_grad():
            max_next_action = self.q_dre(next_states).max(dim=1)[1].view(-1, 1)
            target_q_dre_values = self.target_q_dre(next_states).gather(1, max_next_action)
        q_dre_loss = F.mse_loss(q_dre_values, rewards + self.gamma * target_q_dre_values * (1 - dones))

        qf_loss = qf1_loss + qf2_loss + q_dre_loss
        return qf_loss, min_qf1_loss_, min_qf2_loss_

    def update(self, t) -> Dict:
        # ref: https://github.com/aviralkumar2907/CQL/blob/d67dbe9cf5d2b96e3b462b6146f249b3d6569796/d4rl/rlkit/torch/sac/cql.py#L41
        self.actor.train()
        self.qf1.train()
        self.qf2.train()
        self.q_dre.train()
        states, actions, rewards, next_states, dones, indices, weights = self.prep_minibatch()
        # update critic 
        qf_loss, min_qf1_loss, min_qf2_loss = self.compute_critic_loss(states, actions, rewards, next_states, dones, indices, weights)
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
        if self.with_lagrange:
            # CQL regularization training
            alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0.0, max=1000000.0)
            min_qf1_loss = alpha_prime * (min_qf1_loss.detach() - self.target_action_gap)
            min_qf2_loss = alpha_prime * (min_qf2_loss.detach() - self.target_action_gap)
            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-min_qf1_loss - min_qf2_loss) * 0.5
            alpha_prime_loss.backward()
            self.alpha_prime_optimizer.step()

        # update the target network
        if t % self.target_net_update_freq == 0:
            self.update_target_model(self.target_qf1, self.qf1)
            self.update_target_model(self.target_qf2, self.qf2)

        return {'qf_loss': qf_loss.detach().cpu().item(), 
                'actor_loss': actor_loss.detach().cpu().item(), 
                'alpha_loss': alpha_loss.detach().cpu().item(),
                'alpha_prime_loss': alpha_prime_loss.detach().cpu().item(),
                'alpha': self.log_alpha.exp().cpu().item(),
                'alpha_prime': self.log_alpha_prime.exp().cpu().item(),}


class CQL_BC(CQL):
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
            self.pi_b_kl = config.USE_PI_B_KL
            if self.pi_b_kl:
                if self.num_feats == 87:
                    self.pi_b_model = torch.load('pi_b_models/model_min_max_agg.pth', map_location=torch.device('cpu')).to(self.device)
                elif self.num_feats == 49:
                    self.pi_b_model = torch.load('pi_b_models/model_mean_agg.pth', map_location=torch.device('cpu')).to(self.device)
                else:
                    raise ValueError
                self.pi_b_model.eval()

    def get_behavior(self, 
                     states: torch.Tensor, 
                     actions: torch.Tensor, 
                     action_probs: torch.Tensor) -> Categorical:
        if self.pi_b_kl:
            behavior_logits = self.pi_b_model(states)
            behavior = Categorical(logits=behavior_logits)
        else:
            # assume other action probabilities is 0.01 of behavior policy
            epsilon = 0.01
            behavior_probs = torch.full(action_probs.shape, epsilon, device=self.device)
            behavior_probs.scatter_(1, actions, 1 - epsilon * (self.num_actions - 1))
            behavior = Categorical(probs=behavior_probs)

        return behavior

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
            kl_div = None
        else:
            behavior = self.get_behavior(states, actions, action_probs)
            policy = Categorical(logits=logits)
            nu = torch.clamp(self.log_nu.exp(), min=0.0, max=1000000.0)
            # \nu * (\beta - KL(\pi_\phi(a|s) || \pi_{b}(a|s)))
            kl_div = kl_divergence(behavior, policy).mean()
            # replace infinity of kl divergence to 20
            kl_div[torch.isinf(kl_div)] = 20.0
            bc_loss = nu * (kl_div - self.bc_kl_beta)
        # TODO: use a suitable coefficient of normalization term
        coef = self.actor_lambda / (action_probs * (self.alpha * log_pi - min_qf_values)).abs().mean().detach()
        total_loss = actor_loss * coef + bc_loss
        return total_loss, actor_loss, bc_loss, kl_div, action_probs, log_pi


    def update(self, t) -> Dict:
        self.actor.train()
        self.qf1.train()
        self.qf2.train()
        self.q_dre.train()
        states, actions, rewards, next_states, dones, indices, weights = self.prep_minibatch()
        # update critic 
        qf_loss, min_qf1_loss, min_qf2_loss = self.compute_critic_loss(states, actions, rewards, next_states, dones, indices, weights)
        self.q_optimizer.zero_grad()
        qf_loss.backward()
        if self.is_gradient_clip:
            self.gradient_clip_q()
        self.q_optimizer.step()
        # update actor 
        total_loss, actor_loss, bc_loss, kl_div, action_probs, log_pi = self.compute_actor_loss(states, actions)
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
            loss['alpha'] = self.log_alpha.exp().cpu().item()
        if self.with_lagrange:
            # CQL regularization training
            alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0.0, max=1000000.0)
            min_qf1_loss = alpha_prime * (min_qf1_loss.detach() - self.target_action_gap)
            min_qf2_loss = alpha_prime * (min_qf2_loss.detach() - self.target_action_gap)
            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-min_qf1_loss - min_qf2_loss) * 0.5
            alpha_prime_loss.backward()
            self.alpha_prime_optimizer.step()
            loss['alpha_prime_loss'] = alpha_prime_loss.detach().cpu().item()
            loss['alpha_prime'] = self.log_alpha_prime.exp().cpu().item()
        if self.bc_type == "KL":
            nu = torch.clamp(self.log_nu.exp(), min=0.0, max=1000000.0)
            nu_loss = -nu * (kl_div.detach() - self.bc_kl_beta)
            self.nu_optimizer.zero_grad()
            nu_loss.backward()
            if self.is_gradient_clip:
                self.log_nu.grad.data.clamp_(-1, 1)
            self.nu_optimizer.step()
            loss['nu_loss'] = nu_loss.detach().cpu().item()
            loss['nu'] = nu.detach().cpu().item()
            loss['kl_loss'] = kl_div.detach().cpu().item()
        # update the target network
        if t % self.target_net_update_freq == 0:
            self.update_target_model(self.target_qf1, self.qf1)
            self.update_target_model(self.target_qf2, self.qf2)

        return loss
