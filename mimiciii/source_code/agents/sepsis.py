import torch
import torch.nn.functional as F

from agents.DQN import *
from agents.SAC import *
from agents.CQL import *
from replay_buffer import ExperienceReplayMemory, PrioritizedReplayMemory

class DQN_regularization(DQN):
    def __init__(self, 
                 env: dict, 
                 config: Config, 
                 log_dir='./logs',
                 static_policy=False):
        super().__init__(env, config, log_dir, static_policy)

        # loss regularization term
        self.reg_lambda = config.REG_LAMBDA
        self.reward_threshold = config.REWARD_THRESHOLD

    def compute_loss(self, batch_vars) -> torch.Tensor:
        '''
            loss function = E[Q_double-target - Q_estimate]^2 + lambda * max(|Q_estimate| - Q_threshold, 0)
            Q_double-target = reward + gamma * Q_double-target(next_state, argmax_a(Q(next_state, a)))
        '''
        states, actions, rewards, next_states, dones, indices, weights = batch_vars
        q_values = self.q(states).gather(1, actions)
        with torch.no_grad():
            max_next_action = self.get_max_next_state_action(next_states)
            target_q_values = self.target_q(next_states).gather(1, max_next_action)
            # empirical hack to make the Q values never exceed the threshold - helps learning
            if self.reg_lambda > 0:
                target_q_values[target_q_values > self.reward_threshold] = self.reward_threshold
                target_q_values[target_q_values < -self.reward_threshold] = -self.reward_threshold

        if self.priority_replay:
            diff = (q_values - (rewards + self.gamma * target_q_values * (1 - dones)))
            # updata priorities whether include regularization or not ???
            self.memory.update_priorities(indices, diff.detach().squeeze().abs().cpu().numpy().tolist())
            loss = ((0.5 * diff.pow(2))* weights).mean()
        else:
            loss = F.mse_loss(q_values, rewards + self.gamma * target_q_values * (1 - dones))
        return loss + self.reg_lambda * max(q_values.abs().max() - self.reward_threshold, 0)


class WDQNE(WDQN):
    def __init__(self, 
                 env: dict, 
                 config: Config, 
                 log_dir='./logs',
                 static_policy=False):
        super().__init__(env, config, log_dir, static_policy)
        self.sofa_threshold = config.SOFA_THRESHOLD

    def declare_memory(self):
        dims = (self.num_feats, 1, 1, self.num_feats, 1, 1, 1)
        self.memory = ExperienceReplayMemory(self.experience_replay_size, dims) \
                        if not self.priority_replay \
                        else PrioritizedReplayMemory(self.experience_replay_size, dims, self.priority_alpha, self.priority_beta_start, self.priority_beta_frames, self.device)

    def append_to_replay(self, s, a, r, s_, a_, done, SOFA):
        self.memory.push((s, a, r, s_, a_, done, SOFA))

    def prep_minibatch(self) -> Tuple[torch.Tensor, 
                                      torch.Tensor,
                                      torch.Tensor,
                                      torch.Tensor,
                                      torch.Tensor,
                                      torch.Tensor,
                                      torch.Tensor,
                                      List,
                                      torch.Tensor]:
        states, actions, rewards, next_states, next_actions, dones, SOFAs, indices, weights = \
            self.memory.sample(self.batch_size)

        states = torch.tensor(states, device=self.device, dtype=torch.float)
        actions = torch.tensor(actions, device=self.device, dtype=torch.int64)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float)
        next_states = torch.tensor(next_states, device=self.device, dtype=torch.float)
        next_actions = torch.tensor(next_actions, device=self.device, dtype=torch.int64)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float)
        SOFAs = torch.tensor(SOFAs, device=self.device, dtype=torch.float)

        return states, actions, rewards, next_states, next_actions, dones, SOFAs, indices, weights

    def compute_loss(self, batch_vars) -> torch.Tensor:
        states, actions, rewards, next_states, next_actions, dones, SOFAs, indices, weights = batch_vars
        q_values = self.q(states).gather(1, actions)
        next_q_values = self.q(next_states)
        max_next_actions = torch.argmax(next_q_values, dim=1, keepdim=True)
        with torch.no_grad():
            target_next_q_values = self.target_q(next_states)
        max_target_next_actions = torch.argmax(target_next_q_values, dim=1, keepdim=True)
        target_next_q_values_softmax = F.softmax(target_next_q_values, dim=1)
        sigma = target_next_q_values_softmax.gather(1, max_next_actions)
        phi = target_next_q_values_softmax.gather(1, max_target_next_actions)
        p = phi / (phi + sigma)
        target_q_values = p * target_next_q_values.max(dim=1)[0].view(-1, 1) + (1 - p) * target_next_q_values.gather(1, max_next_actions)
        # expertise
        expert_q_values = target_next_q_values.gather(1, next_actions)
        target_q_values = torch.where(SOFAs < self.sofa_threshold, expert_q_values, target_q_values)

        if self.priority_replay:
            diff = (q_values - (rewards + self.gamma * target_q_values * (1 - dones)))
            self.memory.update_priorities(indices, diff.detach().squeeze().abs().cpu().numpy().tolist())
            loss = ((0.5 * diff.pow(2))* weights).mean()
        else:
            loss = F.mse_loss(q_values, rewards + self.gamma * target_q_values * (1 - dones))
        return loss


class SAC_BC_E(SAC):
    def __init__(self, 
                 env: dict, 
                 config: Config, 
                 log_dir='./logs',
                 static_policy=False) -> None:
        super().__init__(env, config, log_dir, static_policy)
        self.actor_lambda = config.ACTOR_LAMBDA
        self.sofa_threshold = config.SOFA_THRESHOLD
        self.bc_type = config.BC_TYPE
        if self.bc_type == "KL":
            self.bc_kl_beta = config.BC_KL_BETA
            self.log_nu = torch.zeros(1, dtype=torch.float, device=self.device, requires_grad=True)
            self.nu_optimizer = optim.Adam([self.log_nu], lr=self.q_lr, eps=1e-4)

    def declare_memory(self):
        dims = (self.num_feats, 1, 1, self.num_feats, 1, 1)
        self.memory = ExperienceReplayMemory(self.experience_replay_size, dims) \
                        if not self.priority_replay else \
                        PrioritizedReplayMemory(self.experience_replay_size, dims, self.priority_alpha, self.priority_beta_start, self.priority_beta_frames, self.device)

    def append_to_replay(self, s, a, r, s_, done, SOFA):
        self.memory.push((s, a, r, s_, done, SOFA))

    def prep_minibatch(self) -> Tuple[torch.Tensor, 
                                      torch.Tensor,
                                      torch.Tensor,
                                      torch.Tensor,
                                      torch.Tensor,
                                      torch.Tensor,
                                      List,
                                      torch.Tensor]:
        states, actions, rewards, next_states, dones, SOFAs, indices, weights = self.memory.sample(self.batch_size)

        states = torch.tensor(states, device=self.device, dtype=torch.float)
        actions = torch.tensor(actions, device=self.device, dtype=torch.int64)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float)
        next_states = torch.tensor(next_states, device=self.device, dtype=torch.float)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float)
        SOFAs = torch.tensor(SOFAs, device=self.device, dtype=torch.float)

        return states, actions, rewards, next_states, dones, SOFAs, indices, weights

    def compute_actor_loss(self, states, actions, SOFAs) -> Tuple[torch.Tensor,
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
            bc_loss = F.cross_entropy(action_probs, actions.view(-1), reduction='none') 
            bc_loss = (bc_loss * (SOFAs < self.sofa_threshold).to(torch.float).view(-1).detach()).mean()
            kl_div = None
        else:
            # assume other action probabilities is 0.001 of behavior policy
            clin_probs = torch.full(action_probs.shape, 0.001, device=self.device)
            clin_probs.scatter_(1, actions, 1 - 0.001 * (self.num_actions - 1))
            clin = Categorical(probs=clin_probs)
            policy = Categorical(logits=logits)
            nu = torch.clamp(self.log_nu.exp(), min=0.0, max=1000000.0)
            # \nu * (\beta - KL(\pi_\phi(a|s) || \pi_{clin}(a|s)))
            kl_div = kl_divergence(policy, clin)
            kl_threshold = torch.full(kl_div.shape, self.bc_kl_beta, device=self.device) * SOFAs
            bc_loss = nu * ((kl_div - kl_threshold) * (SOFAs < self.sofa_threshold).to(torch.float).view(-1).detach()).mean()

        coef = self.actor_lambda / (action_probs * (self.alpha * log_pi - min_qf_values)).abs().mean().detach()
        total_loss = actor_loss * coef + bc_loss
        return total_loss, actor_loss * coef, bc_loss, kl_div, action_probs, log_pi

    def update(self, t) -> Dict:
        self.actor.train()
        self.qf1.train()
        self.qf2.train()
        self.q_dre.train()
        states, actions, rewards, next_states, dones, SOFAs, indices, weights = self.prep_minibatch()
        # update critic 
        qf_loss = self.compute_critic_loss(states, actions, rewards, next_states, dones, indices, weights)
        self.q_optimizer.zero_grad()
        qf_loss.backward()
        if self.is_gradient_clip:
            self.gradient_clip_q()
        self.q_optimizer.step()
        # update actor 
        total_loss, actor_loss, bc_loss, kl_div, action_probs, log_pi = self.compute_actor_loss(states, actions, SOFAs)
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
            kl_threshold = torch.full(kl_div.shape, self.bc_kl_beta, device=self.device) * SOFAs
            nu_loss = nu * ((kl_div - kl_threshold).detach() * (SOFAs < self.sofa_threshold).to(torch.float).view(-1).detach()).mean()
            self.nu_optimizer.zero_grad()
            nu_loss.backward()
            self.nu_optimizer.step()
            loss['nu_loss'] = nu_loss.detach().cpu().item()
            loss['kl_loss'] = kl_div.mean().detach().cpu().item()
        # update the target network
        if t % self.target_net_update_freq == 0:
            self.update_target_model(self.target_qf1, self.qf1)
            self.update_target_model(self.target_qf2, self.qf2)

        return loss


class CQL_BC_E(CQL):
    def __init__(self, 
                 env: dict, 
                 config: Config, 
                 log_dir='./logs',
                 static_policy=False) -> None:
        super().__init__(env, config, log_dir, static_policy)
        self.actor_lambda = config.ACTOR_LAMBDA
        self.sofa_threshold = config.SOFA_THRESHOLD
        self.bc_type = config.BC_TYPE
        if self.bc_type == "KL":
            self.bc_kl_beta = config.BC_KL_BETA
            self.log_nu = torch.zeros(1, dtype=torch.float, device=self.device, requires_grad=True)
            self.nu_optimizer = optim.Adam([self.log_nu], lr=self.q_lr, eps=1e-4)

    def declare_memory(self):
        dims = (self.num_feats, 1, 1, self.num_feats, 1, 1)
        self.memory = ExperienceReplayMemory(self.experience_replay_size, dims) \
                        if not self.priority_replay else \
                        PrioritizedReplayMemory(self.experience_replay_size, dims, self.priority_alpha, self.priority_beta_start, self.priority_beta_frames, self.device)

    def append_to_replay(self, s, a, r, s_, done, SOFA):
        self.memory.push((s, a, r, s_, done, SOFA))

    def prep_minibatch(self) -> Tuple[torch.Tensor, 
                                      torch.Tensor,
                                      torch.Tensor,
                                      torch.Tensor,
                                      torch.Tensor,
                                      torch.Tensor,
                                      List,
                                      torch.Tensor]:
        states, actions, rewards, next_states, dones, SOFAs, indices, weights = self.memory.sample(self.batch_size)

        states = torch.tensor(states, device=self.device, dtype=torch.float)
        actions = torch.tensor(actions, device=self.device, dtype=torch.int64)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float)
        next_states = torch.tensor(next_states, device=self.device, dtype=torch.float)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float)
        SOFAs = torch.tensor(SOFAs, device=self.device, dtype=torch.float)

        return states, actions, rewards, next_states, dones, SOFAs, indices, weights

    def compute_actor_loss(self, states, actions, SOFAs) -> Tuple[torch.Tensor,
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
            bc_loss = F.cross_entropy(action_probs, actions.view(-1), reduction='none') 
            bc_loss = (bc_loss * (SOFAs < self.sofa_threshold).to(torch.float).view(-1).detach()).mean()
        else:
            # assume other action probabilities is 0.001 of behavior policy
            clin_probs = torch.full(action_probs.shape, 0.001, device=self.device)
            clin_probs.scatter_(1, actions, 1 - 0.001 * (self.num_actions - 1))
            clin = Categorical(probs=clin_probs)
            policy = Categorical(logits=logits)
            nu = torch.clamp(self.log_nu.exp(), min=0.0, max=1000000.0)
            # \nu * (\beta - KL(\pi_\phi(a|s) || \pi_{clin}(a|s)))
            kl_div = kl_divergence(policy, clin)
            kl_threshold = torch.full(kl_div.shape, self.bc_kl_beta, device=self.device) * SOFAs
            bc_loss = nu * ((kl_div - kl_threshold) * (SOFAs < self.sofa_threshold).to(torch.float).view(-1).detach()).mean()

        coef = self.actor_lambda / (action_probs * (self.alpha * log_pi - min_qf_values)).abs().mean().detach()
        total_loss = actor_loss * coef + bc_loss
        return total_loss, actor_loss * coef, bc_loss, kl_div, action_probs, log_pi

    def update(self, t) -> Dict:
        self.actor.train()
        self.qf1.train()
        self.qf2.train()
        self.q_dre.train()
        states, actions, rewards, next_states, dones, SOFAs, indices, weights = self.prep_minibatch()
        # update critic 
        qf_loss, min_qf1_loss, min_qf2_loss = self.compute_critic_loss(states, actions, rewards, next_states, dones, indices, weights)
        self.q_optimizer.zero_grad()
        qf_loss.backward()
        if self.is_gradient_clip:
            self.gradient_clip_q()
        self.q_optimizer.step()
        # update actor 
        total_loss, actor_loss, bc_loss, kl_div, action_probs, log_pi = self.compute_actor_loss(states, actions, SOFAs)
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
        if self.bc_type == "KL":
            nu = torch.clamp(self.log_nu.exp(), min=0.0, max=1000000.0)
            kl_threshold = torch.full(kl_div.shape, self.bc_kl_beta, device=self.device) * SOFAs
            nu_loss = nu * ((kl_div - kl_threshold).detach() * (SOFAs < self.sofa_threshold).to(torch.float).view(-1).detach()).mean()
            self.nu_optimizer.zero_grad()
            nu_loss.backward()
            self.nu_optimizer.step()
            loss['nu_loss'] = nu_loss.detach().cpu().item()
            loss['kl_loss'] = kl_div.mean().detach().cpu().item()
        # update the target network
        if t % self.target_net_update_freq == 0:
            self.update_target_model(self.target_qf1, self.qf1)
            self.update_target_model(self.target_qf2, self.qf2)

        return loss
