import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, kl_divergence
import os

from agents.SAC import SAC, SAC_BC
from agents.BC import BC, MAX_KL_DIV
from utils import Config

class CQL(SAC):
    def __init__(
        self, 
        env: dict, 
        config: Config, 
        log_dir: str = './logs',
        static_policy: bool = False
    ) -> None:
        super().__init__(config=config, env=env, log_dir=log_dir, static_policy=static_policy)
        # TODO: delete q_dre 
        self.with_lagrange = config.WITH_LAGRANGE
        if self.with_lagrange:
            self.target_action_gap = config.TARGET_ACTION_GAP
            self.log_alpha_prime = torch.zeros(1, dtype=torch.float, device=self.device, requires_grad=True)
            self.alpha_prime_optimizer = optim.Adam([self.log_alpha_prime], lr=self.q_lr, eps=1e-4)

    def save_checkpoint(self, epoch: int, name: str = 'checkpoint.pth'):
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

    def load_checkpoint(self, name: str = 'checkpoint.pth') -> int:
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

    def compute_critic_loss(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor, 
        rewards: torch.Tensor, 
        next_states: torch.Tensor, 
        dones: torch.Tensor, 
        indices: list, 
        weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_state_augmentation:
            states, next_states = self.augmentation(states, next_states, rewards, dones)
            actions = actions.unsqueeze(1).repeat(1, 2, 1)

        qf1_values = self.qf1(states).gather(-1, actions)
        qf2_values = self.qf2(states).gather(-1, actions)

        # CQL regulariztion loss
        min_qf1_loss_ = torch.logsumexp(self.qf1(states), dim=-1).mean() - qf1_values.mean()
        min_qf2_loss_ = torch.logsumexp(self.qf2(states), dim=-1).mean() - qf2_values.mean()

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
            min_qf_next_target = min_qf_next_target.sum(dim=-1, keepdim=True)

        if self.use_state_augmentation:
            qf1_values = qf1_values.mean(dim=1)
            qf2_values = qf2_values.mean(dim=1)
            min_qf_next_target = min_qf_next_target.mean(dim=1)

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

        qf_loss = qf1_loss + qf2_loss
        return qf_loss, min_qf1_loss_, min_qf2_loss_

    def update_alpha_prime(self, min_qf1_loss: torch.Tensor, min_qf2_loss: torch.Tensor) -> dict[str, float]:
        # CQL regularization training
        alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0.0, max=1000000.0)
        min_qf1_loss = alpha_prime * (min_qf1_loss.detach() - self.target_action_gap)
        min_qf2_loss = alpha_prime * (min_qf2_loss.detach() - self.target_action_gap)
        self.alpha_prime_optimizer.zero_grad()
        alpha_prime_loss = (-min_qf1_loss - min_qf2_loss) * 0.5
        alpha_prime_loss.backward()
        self.alpha_prime_optimizer.step()
        return {
            'alpha_prime_loss': alpha_prime_loss.detach().cpu().item(),
            'alpha_prime': self.log_alpha_prime.exp().cpu().item()
        }

    def update(self, t: int) -> dict[str, int]:
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

        loss = {
            'qf_loss': qf_loss.detach().cpu().item(), 
            'actor_loss': actor_loss.detach().cpu().item()
        }

        if self.autotune:
            alpha_loss = self.update_alpha(action_probs, log_pi)
            loss.update(alpha_loss)
        if self.with_lagrange:
            alpha_prime_loss = self.update_alpha_prime(min_qf1_loss, min_qf2_loss)
            loss.update(alpha_prime_loss)
        # update the target network
        if t % self.target_net_update_freq == 0:
            self.update_target_model(self.target_qf1, self.qf1)
            self.update_target_model(self.target_qf2, self.qf2)

        return loss


class CQL_BC(CQL, SAC_BC):
    def __init__(
        self, 
        env: dict, 
        config: Config, 
        log_dir: str = './logs',
        static_policy: bool = False
    ) -> None:
        CQL.__init__(self, config=config, env=env, log_dir=log_dir, static_policy=static_policy)
        SAC_BC.__init__(self, config=config, env=env, log_dir=log_dir, static_policy=static_policy)

        self.actor_lambda = config.ACTOR_LAMBDA

    def save_checkpoint(self, epoch: int, name: str = 'checkpoint.pth'):
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
        if self.bc_type == "KL":
            checkpoint['log_nu'] = self.log_nu.item()
            checkpoint['nu_optimizer'] = self.nu_optimizer.state_dict()
        torch.save(checkpoint, os.path.join(self.log_dir, name))

    def load_checkpoint(self, name: str = 'checkpoint.pth') -> int:
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
        if self.bc_type == "KL":
            self.log_nu = torch.tensor(np.array([checkpoint['log_nu']]), dtype=torch.float, requires_grad=True, device=self.device)
            self.nu_optimizer.load_state_dict = checkpoint['nu_optimizer']
            self.nu = self.log_nu.exp().item()
        return checkpoint['epoch']

    def update(self, t: int) -> dict[str, int]:
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

        loss = {
            'qf_loss': qf_loss.detach().cpu().item(), 
            'actor_loss': actor_loss.detach().cpu().item(), 
            'bc_loss': bc_loss.detach().cpu().item()
        }

        if self.autotune:
            alpha_loss = self.update_alpha(action_probs, log_pi)
            loss.update(alpha_loss)
        if self.with_lagrange:
            alpha_prime_loss = self.update_alpha_prime(min_qf1_loss, min_qf2_loss)
            loss.update(alpha_prime_loss)
        if self.bc_type == "KL":
            nu_loss = self.update_nu(kl_div, self.is_gradient_clip)
            loss.update(nu_loss)
            loss['kl_div'] = kl_div.detach().cpu().item()
        # update the target network
        if t % self.target_net_update_freq == 0:
            self.update_target_model(self.target_qf1, self.qf1)
            self.update_target_model(self.target_qf2, self.qf2)

        return loss
