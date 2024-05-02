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
            loss = (F.smooth_l1_loss(q_values, 
                                    rewards + self.gamma * target_q_values * (1 - dones),
                                    reduction='none') * weights).mean()
        else:
            loss = F.smooth_l1_loss(q_values, 
                                    rewards + self.gamma * target_q_values * (1 - dones))
        return loss


class SAC_BC_E(SAC_BC):
    def __init__(self, 
                 env: dict, 
                 config: Config, 
                 log_dir='./logs',
                 static_policy=False) -> None:
        super().__init__(env, config, log_dir, static_policy)

        self.sofa_threshold = config.SOFA_THRESHOLD
        self.use_sofa_cv = config.USE_SOFA_CV
        self.is_sofa_threshold_below = config.IS_SOFA_THRESHOLD_BELOW
        self.kl_threshold_type = config.KL_THRESHOLD_TYPE
        self.kl_threshold_exp = config.KL_THRESHOLD_EXP
        self.kl_threshold_coef = config.KL_THRESHOLD_COEF

    def declare_memory(self):
        dims = (self.num_feats, 1, 1, self.num_feats, 1, 1, 1)
        if self.priority_replay:
            self.memory = PrioritizedReplayMemory(self.experience_replay_size, 
                                                  dims, 
                                                  self.priority_alpha, 
                                                  self.priority_beta_start, 
                                                  self.priority_beta_frames, 
                                                  self.device)
        else:
            self.memory = ExperienceReplayMemory(self.experience_replay_size, dims)
                        

    def append_to_replay(self, s, a, r, s_, done, SOFA, SOFA_CV):
        self.memory.push((s, a, r, s_, done, SOFA, SOFA_CV))

    def prep_minibatch(self) -> Tuple[torch.Tensor, 
                                      torch.Tensor,
                                      torch.Tensor,
                                      torch.Tensor,
                                      torch.Tensor,
                                      torch.Tensor,
                                      torch.Tensor,
                                      List,
                                      torch.Tensor]:
        states, actions, rewards, next_states, dones, SOFAs, SOFA_CVs, indices, weights = \
            self.memory.sample(self.batch_size)

        states = torch.tensor(states, device=self.device, dtype=torch.float)
        actions = torch.tensor(actions, device=self.device, dtype=torch.int64)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float)
        next_states = torch.tensor(next_states, device=self.device, dtype=torch.float)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float)
        SOFAs = torch.tensor(SOFAs, device=self.device, dtype=torch.float)
        SOFA_CVs = torch.tensor(SOFA_CVs, device=self.device, dtype=torch.float)

        return states, actions, rewards, next_states, dones, SOFAs, SOFA_CVs, indices, weights

    def compute_kl_threshold(self, shape, bc_condition: torch.Tensor) -> torch.Tensor:
        if self.kl_threshold_type == 'step':
            # add 1 to avoid threshold be 0
            kl_threshold = torch.full(shape, self.bc_kl_beta, device=self.device) * (6 - bc_condition.view(-1,))
        elif self.kl_threshold_type == 'exp':
            kl_threshold = self.kl_threshold_coef * \
                            (torch.full(shape, self.kl_threshold_exp, device=self.device) ** bc_condition.view(-1,))
        else:
            raise ValueError("Wrong kl threshold type!")
        return kl_threshold.detach()

    def compute_bc_loss(self, 
                        kl_div: torch.Tensor, 
                        kl_threshold: torch.Tensor, 
                        bc_condition: torch.Tensor) -> torch.Tensor:
        # \nu * (KL(\pi_\phi(a|s) || \pi_{clin}(a|s)) - \beta)
        nu = torch.clamp(self.log_nu.exp(), min=0.0, max=1000000.0)
        if self.is_sofa_threshold_below:
            mask = (bc_condition < self.sofa_threshold).to(torch.float).view(-1).detach()
        else:
            mask = (bc_condition >= self.sofa_threshold).to(torch.float).view(-1).detach()

        bc_loss = nu * ((kl_div - kl_threshold.detach()) * mask).mean()

        return bc_loss

    def compute_actor_loss(self, 
                           states: torch.Tensor, 
                           actions: torch.Tensor, 
                           bc_condition: torch.Tensor) -> Tuple[torch.Tensor,
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
            # TODO: fix this
            bc_loss = F.cross_entropy(action_probs, actions.view(-1), reduction='none') 
            bc_loss = (bc_loss * (bc_condition < self.sofa_threshold).to(torch.float).view(-1).detach()).mean()
            kl_div = None
        else:
            clin = self.get_behavior(states, actions, action_probs)
            policy = Categorical(logits=logits)
            kl_div = kl_divergence(clin, policy)
            # replace infinity of kl divergence to 20
            kl_div[torch.isinf(kl_div)] = 20.0
            kl_threshold = self.compute_kl_threshold(kl_div.shape, bc_condition)
            bc_loss = self.compute_bc_loss(kl_div, kl_threshold, bc_condition)

        coef = self.actor_lambda / (action_probs * (self.alpha * log_pi - min_qf_values)).abs().mean().detach()
        total_loss = actor_loss * coef + bc_loss
        return total_loss, actor_loss, bc_loss, kl_div, kl_threshold, action_probs, log_pi

    def update(self, t) -> Dict:
        self.actor.train()
        self.qf1.train()
        self.qf2.train()
        self.q_dre.train()
        states, actions, rewards, next_states, dones, SOFAs, SOFA_CVs, indices, weights = self.prep_minibatch()
        # update critic 
        qf_loss = self.compute_critic_loss(states, actions, rewards, next_states, dones, indices, weights)
        self.q_optimizer.zero_grad()
        qf_loss.backward()
        if self.is_gradient_clip:
            self.gradient_clip_q()
        self.q_optimizer.step()
        # update actor 
        bc_condition = SOFA_CVs if self.use_sofa_cv else SOFAs
        total_loss, actor_loss, bc_loss, kl_div, kl_threshold, action_probs, log_pi = \
                                            self.compute_actor_loss(states, actions, bc_condition)
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
            alpha = self.log_alpha.exp()
            alpha_loss = (action_probs.detach() * (-alpha * (log_pi + self.target_entropy).detach())).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
            loss['alpha_loss'] = alpha_loss.detach().cpu().item()
            loss['alpha'] = alpha.detach().cpu().item()
        if self.bc_type == "KL":
            nu = torch.clamp(self.log_nu.exp(), min=0.0, max=1000000.0)
            if self.is_sofa_threshold_below:
                mask = (bc_condition < self.sofa_threshold).to(torch.float).view(-1).detach()
            else:
                mask = (bc_condition >= self.sofa_threshold).to(torch.float).view(-1).detach()
            nu_loss = -nu * ((kl_div - kl_threshold).detach() * mask).mean()
            self.nu_optimizer.zero_grad()
            nu_loss.backward()
            if self.is_gradient_clip:
                self.log_nu.grad.data.clamp_(-1, 1)
            self.nu_optimizer.step()
            loss['nu_loss'] = nu_loss.detach().cpu().item()
            loss['kl_div'] = kl_div[mask.to(torch.bool)].mean().detach().cpu().item()
            loss['kl_threshold'] = kl_threshold[mask.to(torch.bool)].mean().cpu().item()
            loss['nu'] = nu.detach().cpu().item()
        # update the target network
        if t % self.target_net_update_freq == 0:
            self.update_target_model(self.target_qf1, self.qf1)
            self.update_target_model(self.target_qf2, self.qf2)

        return loss


class CQL_BC_E(CQL_BC):
    def __init__(self, 
                 env: dict, 
                 config: Config, 
                 log_dir='./logs',
                 static_policy=False) -> None:
        super().__init__(env, config, log_dir, static_policy)

        self.sofa_threshold = config.SOFA_THRESHOLD
        self.is_sofa_threshold_below = config.IS_SOFA_THRESHOLD_BELOW
        self.use_sofa_cv = config.USE_SOFA_CV
        self.kl_threshold_type = config.KL_THRESHOLD_TYPE
        self.kl_threshold_exp = config.KL_THRESHOLD_EXP
        self.kl_threshold_coef = config.KL_THRESHOLD_COEF

    def declare_memory(self):
        dims = (self.num_feats, 1, 1, self.num_feats, 1, 1, 1)
        if self.priority_replay:
            self.memory = PrioritizedReplayMemory(self.experience_replay_size, 
                                                  dims, 
                                                  self.priority_alpha, 
                                                  self.priority_beta_start, 
                                                  self.priority_beta_frames, 
                                                  self.device)
        else:
            self.memory = ExperienceReplayMemory(self.experience_replay_size, dims)
                        

    def append_to_replay(self, s, a, r, s_, done, SOFA, SOFA_CV):
        self.memory.push((s, a, r, s_, done, SOFA, SOFA_CV))

    def prep_minibatch(self) -> Tuple[torch.Tensor, 
                                      torch.Tensor,
                                      torch.Tensor,
                                      torch.Tensor,
                                      torch.Tensor,
                                      torch.Tensor,
                                      torch.Tensor,
                                      List,
                                      torch.Tensor]:
        states, actions, rewards, next_states, dones, SOFAs, SOFA_CVs, indices, weights = \
                                                                self.memory.sample(self.batch_size)

        states = torch.tensor(states, device=self.device, dtype=torch.float)
        actions = torch.tensor(actions, device=self.device, dtype=torch.int64)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float)
        next_states = torch.tensor(next_states, device=self.device, dtype=torch.float)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float)
        SOFAs = torch.tensor(SOFAs, device=self.device, dtype=torch.float)
        SOFA_CVs = torch.tensor(SOFA_CVs, device=self.device, dtype=torch.float)

        return states, actions, rewards, next_states, dones, SOFAs, SOFA_CVs, indices, weights

    def compute_kl_threshold(self, shape, bc_condition: torch.Tensor) -> torch.Tensor:
        if self.kl_threshold_type == 'step':
            # add 1 to avoid threshold be 0
            kl_threshold = torch.full(shape, self.bc_kl_beta, device=self.device) * (6 - bc_condition.view(-1))
        elif self.kl_threshold_type == 'exp':
            kl_threshold = self.kl_threshold_coef * \
                            (torch.full(shape, self.kl_threshold_exp, device=self.device) ** bc_condition.view(-1))
        else:
            raise ValueError("Wrong kl threshold type!")
        return kl_threshold.detach()

    def compute_bc_loss(self, 
                        kl_div: torch.Tensor, 
                        kl_threshold: torch.Tensor, 
                        bc_condition: torch.Tensor) -> torch.Tensor:
        # \nu * (KL(\pi_\phi(a|s) || \pi_{clin}(a|s)) - \beta)
        nu = torch.clamp(self.log_nu.exp(), min=0.0, max=1000000.0)
        if self.is_sofa_threshold_below:
            mask = (bc_condition < self.sofa_threshold).to(torch.float).view(-1).detach()
        else:
            mask = (bc_condition >= self.sofa_threshold).to(torch.float).view(-1).detach()

        bc_loss = nu * ((kl_div - kl_threshold) * mask).mean()

        return bc_loss

    def compute_actor_loss(self, 
                           states: torch.Tensor, 
                           actions: torch.Tensor, 
                           bc_condition: torch.Tensor) -> Tuple[torch.Tensor,
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
            # TODO: fix this
            bc_loss = F.cross_entropy(action_probs, actions.view(-1), reduction='none') 
            bc_loss = (bc_loss * (bc_condition < self.sofa_threshold).to(torch.float).view(-1).detach()).mean()
            kl_div = None
        else:
            clin = self.get_behavior(states, actions, action_probs)
            policy = Categorical(logits=logits)
            kl_div = kl_divergence(clin, policy)
            # replace infinity of kl divergence to 20
            kl_div[torch.isinf(kl_div)] = 20.0
            kl_threshold = self.compute_kl_threshold(kl_div.shape, bc_condition)
            bc_loss = self.compute_bc_loss(kl_div, kl_threshold, bc_condition)

        coef = self.actor_lambda / (action_probs * (self.alpha * log_pi - min_qf_values)).abs().mean().detach()
        total_loss = actor_loss * coef + bc_loss
        return total_loss, actor_loss, bc_loss, kl_div, action_probs, log_pi

    def update(self, t) -> Dict:
        self.actor.train()
        self.qf1.train()
        self.qf2.train()
        self.q_dre.train()
        states, actions, rewards, next_states, dones, SOFAs, SOFA_CVs, indices, weights = self.prep_minibatch()
        # update critic 
        qf_loss, min_qf1_loss, min_qf2_loss = self.compute_critic_loss(states, actions, rewards, next_states, dones, indices, weights)
        self.q_optimizer.zero_grad()
        qf_loss.backward()
        if self.is_gradient_clip:
            self.gradient_clip_q()
        self.q_optimizer.step()
        # update actor 
        bc_condition = SOFA_CVs if self.use_sofa_cv else SOFAs
        total_loss, actor_loss, bc_loss, kl_div, action_probs, log_pi = \
                                            self.compute_actor_loss(states, actions, bc_condition)
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
            alpha = self.log_alpha.exp()
            alpha_loss = (action_probs.detach() * (-alpha * (log_pi + self.target_entropy).detach())).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
            loss['alpha_loss'] = alpha_loss.detach().cpu().item()
            loss['alpha'] = alpha.detach().cpu().item()
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
            loss['alpha_prime'] = alpha_prime.detach().cpu().item()
        if self.bc_type == "KL":
            nu = torch.clamp(self.log_nu.exp(), min=0.0, max=1000000.0)
            kl_threshold = self.compute_kl_threshold(kl_div.shape, bc_condition)
            if self.is_sofa_threshold_below:
                mask = (bc_condition < self.sofa_threshold).to(torch.float).view(-1).detach()
            else:
                mask = (bc_condition >= self.sofa_threshold).to(torch.float).view(-1).detach()
            nu_loss = -nu * ((kl_div - kl_threshold).detach() * mask).mean()
            if self.is_gradient_clip:
                self.log_nu.grad.data.clamp_(-1, 1)
            self.nu_optimizer.zero_grad()
            nu_loss.backward()
            self.nu_optimizer.step()
            loss['nu_loss'] = nu_loss.detach().cpu().item()
            loss['kl_div'] = kl_div[mask.to(torch.bool)].mean().detach().cpu().item()
            loss['kl_threshold'] = kl_threshold[mask.to(torch.bool)].mean().cpu().item()
            loss['nu'] = nu.detach().cpu().item()
        # update the target network
        if t % self.target_net_update_freq == 0:
            self.update_target_model(self.target_qf1, self.qf1)
            self.update_target_model(self.target_qf2, self.qf2)

        return loss
