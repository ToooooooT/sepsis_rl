import torch
import torch.nn.functional as F

from agents.DQN import *
from agents.SAC import *
from agents.CQL import *
from agents.BC import BCE, MAX_KL_DIV
from replay_buffer import ExperienceReplayMemory, PrioritizedReplayMemory

class DQN_regularization(DQN):
    def __init__(
        self, 
        env: dict, 
        config: Config, 
        log_dir: str = './logs',
        static_policy: bool = False
    ):
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

        if self.use_state_augmentation:
            states, next_states = self.augmentation(states, next_states, rewards, dones)
            actions = actions.unsqueeze(1).repeat(1, 2, 1)

        q_values = self.q(states).gather(-1, actions)
        with torch.no_grad():
            max_next_action = self.get_max_next_state_action(next_states)
            target_q_values = self.target_q(next_states).gather(-1, max_next_action)
            # empirical hack to make the Q values never exceed the threshold - helps learning
            if self.reg_lambda > 0:
                target_q_values[target_q_values > self.reward_threshold] = self.reward_threshold
                target_q_values[target_q_values < -self.reward_threshold] = -self.reward_threshold

        if self.use_state_augmentation:
            q_values = q_values.mean(dim=1)
            target_q_values = target_q_values.mean(dim=1)

        if self.priority_replay:
            diff = (q_values - (rewards + self.gamma * target_q_values * (1 - dones)))
            # updata priorities whether include regularization or not ???
            self.memory.update_priorities(indices, diff.detach().squeeze().abs().cpu().numpy().tolist())
            loss = ((0.5 * diff.pow(2))* weights).mean()
        else:
            loss = F.mse_loss(q_values, rewards + self.gamma * target_q_values * (1 - dones))
        return loss + self.reg_lambda * max(q_values.abs().max() - self.reward_threshold, 0)


class WDQNE(WDQN):
    def __init__(
        self, 
        env: dict, 
        config: Config, 
        log_dir: str = './logs',
        static_policy: bool = False
    ):
        super().__init__(env, config, log_dir, static_policy)
        self.sofa_threshold = config.SOFA_THRESHOLD


    def append_to_replay(
        self, 
        s: np.ndarray,
        a: np.ndarray,
        r: np.ndarray,
        s_: np.ndarray,
        a_: np.ndarray,
        done: np.ndarray,
        sofa: np.ndarray
    ):
        self.memory.push((s, a, r, s_, a_, done, sofa))

    def prep_minibatch(self) -> tuple[torch.Tensor, 
                                      torch.Tensor,
                                      torch.Tensor,
                                      torch.Tensor,
                                      torch.Tensor,
                                      torch.Tensor,
                                      torch.Tensor,
                                      list,
                                      torch.Tensor]:
        states, actions, rewards, next_states, next_actions, dones, sofas, indices, weights = \
            self.memory.sample(self.batch_size)

        states = torch.tensor(states, device=self.device, dtype=torch.float)
        actions = torch.tensor(actions, device=self.device, dtype=torch.int64)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float)
        next_states = torch.tensor(next_states, device=self.device, dtype=torch.float)
        next_actions = torch.tensor(next_actions, device=self.device, dtype=torch.int64)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float)
        sofas = torch.tensor(sofas, device=self.device, dtype=torch.float)

        return states, actions, rewards, next_states, next_actions, dones, sofas, indices, weights

    def compute_loss(self, batch_vars) -> torch.Tensor:
        states, actions, rewards, next_states, next_actions, dones, sofa, indices, weights = batch_vars

        if self.use_state_augmentation:
            states, next_states = self.augmentation(states, next_states, rewards, dones)
            actions = actions.unsqueeze(1).repeat(1, 2, 1)
            next_actions = next_actions.unsqueeze(1).repeat(1, 2, 1)
            sofa = sofa.unsqueeze(1).repeat(1, 2, 1)

        q_values = self.q(states).gather(-1, actions)
        next_q_values = self.q(next_states)
        max_next_actions = torch.argmax(next_q_values, dim=-1, keepdim=True)
        with torch.no_grad():
            target_next_q_values = self.target_q(next_states)
        max_target_next_actions = torch.argmax(target_next_q_values, dim=-1, keepdim=True)
        target_next_q_values_softmax = F.softmax(target_next_q_values, dim=-1)
        sigma = target_next_q_values_softmax.gather(-1, max_next_actions)
        phi = target_next_q_values_softmax.gather(-1, max_target_next_actions)
        p = phi / (phi + sigma)
        target_q_values = p * target_next_q_values.max(dim=-1, keepdim=True)[0] \
                            + (1 - p) * target_next_q_values.gather(-1, max_next_actions)
        # expertise
        expert_q_values = target_next_q_values.gather(-1, next_actions)
        target_q_values = torch.where(sofa < self.sofa_threshold, expert_q_values, target_q_values)

        if self.use_state_augmentation:
            q_values = q_values.mean(dim=1)
            target_q_values = target_q_values.mean(dim=1)

        if self.priority_replay:
            diff = (q_values - (rewards + self.gamma * target_q_values * (1 - dones)))
            self.memory.update_priorities(indices, diff.detach().squeeze().abs().cpu().numpy().tolist())
            loss = (F.smooth_l1_loss(q_values, 
                                    rewards + self.gamma * target_q_values * (1 - dones),
                                    reduction='none') * weights).mean()
        else:
            loss = F.smooth_l1_loss(q_values, rewards + self.gamma * target_q_values * (1 - dones))
        return loss


class SAC_BC_E(SAC_BC, BCE):
    def __init__(
        self, 
        env: dict, 
        config: Config, 
        log_dir: str = './logs',
        static_policy: bool = False
    ) -> None:
        SAC_BC.__init__(self, env, config, log_dir, static_policy)
        BCE.__init__(
            self, 
            device=self.device,
            bc_type=config.BC_TYPE,
            bc_kl_beta=config.BC_KL_BETA,
            use_pi_b_kl=config.USE_PI_B_KL,
            nu_lr=self.q_lr,
            phy_epsilon=config.PHY_EPSILON,
            sofa_threshold=config.SOFA_THRESHOLD,
            use_sofa_cv=config.USE_SOFA_CV,
            is_sofa_threshold_below=config.IS_SOFA_THRESHOLD_BELOW,
            kl_threshold_type=config.KL_THRESHOLD_TYPE,
            kl_threshold_exp=config.KL_THRESHOLD_EXP,
            kl_threshold_coef=config.KL_THRESHOLD_COEF,
        )

    def append_to_replay(
        self, 
        s: np.ndarray,
        a: np.ndarray,
        r: np.ndarray,
        s_: np.ndarray,
        done: np.ndarray,
        sofa: np.ndarray,
        sofa_cv: np.ndarray
    ):
        self.memory.push((s, a, r, s_, done, sofa, sofa_cv))

    def prep_minibatch(self) -> tuple[torch.Tensor, 
                                      torch.Tensor,
                                      torch.Tensor,
                                      torch.Tensor,
                                      torch.Tensor,
                                      torch.Tensor,
                                      torch.Tensor,
                                      list,
                                      torch.Tensor]:
        states, actions, rewards, next_states, dones, sofas, sofa_cvs, indices, weights = \
            self.memory.sample(self.batch_size)

        states = torch.tensor(states, device=self.device, dtype=torch.float)
        actions = torch.tensor(actions, device=self.device, dtype=torch.int64)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float)
        next_states = torch.tensor(next_states, device=self.device, dtype=torch.float)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float)
        sofas = torch.tensor(sofas, device=self.device, dtype=torch.float)
        sofa_cvs = torch.tensor(sofa_cvs, device=self.device, dtype=torch.float)

        return states, actions, rewards, next_states, dones, sofas, sofa_cvs, indices, weights

    def compute_actor_loss(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor, 
        bc_condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _, logits, log_pi, action_probs = self.get_action_probs(states)
        with torch.no_grad():
            qf1_values = self.qf1(states)
            qf2_values = self.qf2(states)
            min_qf_values = torch.min(qf1_values, qf2_values)
        # no need for reparameterization, the expectation can be calculated for discrete actions
        actor_loss = (action_probs * (self.alpha * log_pi - min_qf_values)).mean()

        if self.bc_type == 'cross_entropy':
            bc_loss = F.cross_entropy(action_probs, actions.view(-1), reduction='none') 
            mask = self.get_mask(bc_condition)
            bc_loss = (bc_loss * mask).mean()
            kl_div = None
            kl_threshold = None
        else:
            clin = self.get_behavior(states, actions, action_probs, self.device)
            policy = Categorical(logits=logits)
            kl_div = kl_divergence(clin, policy)
            # replace infinity of kl divergence to 20
            kl_div[torch.isinf(kl_div)] = MAX_KL_DIV
            kl_threshold = self.compute_kl_threshold(kl_div.shape, bc_condition, self.device)
            bc_loss = self.compute_bc_loss(kl_div, kl_threshold, bc_condition)

        coef = self.actor_lambda / (action_probs * (self.alpha * log_pi - min_qf_values)).abs().mean().detach()
        total_loss = actor_loss * coef + bc_loss
        return total_loss, actor_loss, bc_loss, kl_div, kl_threshold, action_probs, log_pi

    def update(self, t: int) -> dict[str, int]:
        self.actor.train()
        self.qf1.train()
        self.qf2.train()
        self.q_dre.train()
        states, actions, rewards, next_states, dones, sofas, sofa_cvs, indices, weights = self.prep_minibatch()
        # update critic 
        qf_loss = self.compute_critic_loss(states, actions, rewards, next_states, dones, indices, weights)
        self.q_optimizer.zero_grad()
        qf_loss.backward()
        if self.is_gradient_clip:
            self.gradient_clip_q()
        self.q_optimizer.step()
        # update actor 
        bc_condition = sofa_cvs if self.use_sofa_cv else sofas
        total_loss, actor_loss, bc_loss, kl_div, kl_threshold, action_probs, log_pi = \
                                            self.compute_actor_loss(states, actions, bc_condition)
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
        if self.bc_type == "KL":
            nu_loss = self.update_nu(kl_div, kl_threshold, bc_condition, self.is_gradient_clip)
            loss.update(nu_loss)
            mask = self.get_mask(bc_condition)
            loss['kl_div'] = kl_div[mask.to(torch.bool)].mean().detach().cpu().item()
            loss['kl_threshold'] = kl_threshold[mask.to(torch.bool)].mean().cpu().item()
        # update the target network
        if t % self.target_net_update_freq == 0:
            self.update_target_model(self.target_qf1, self.qf1)
            self.update_target_model(self.target_qf2, self.qf2)

        return loss


class CQL_BC_E(CQL_BC, SAC_BC_E):
    def __init__(
        self, 
        env: dict, 
        config: Config, 
        log_dir: str = './logs',
        static_policy: bool = False
    ) -> None:
        CQL_BC.__init__(self, env, config, log_dir, static_policy)
        SAC_BC_E.__init__(self, env, config, log_dir, static_policy)

    def update(self, t: int) -> dict[str, int]:
        self.actor.train()
        self.qf1.train()
        self.qf2.train()
        self.q_dre.train()
        states, actions, rewards, next_states, dones, sofas, sofa_cvs, indices, weights = self.prep_minibatch()
        # update critic 
        qf_loss, min_qf1_loss, min_qf2_loss = self.compute_critic_loss(states, actions, rewards, next_states, dones, indices, weights)
        self.q_optimizer.zero_grad()
        qf_loss.backward()
        if self.is_gradient_clip:
            self.gradient_clip_q()
        self.q_optimizer.step()
        # update actor 
        bc_condition = sofa_cvs if self.use_sofa_cv else sofas
        total_loss, actor_loss, bc_loss, kl_div, kl_threshold, action_probs, log_pi = \
                                            self.compute_actor_loss(states, actions, bc_condition)
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
            nu_loss = self.update_nu(kl_div, kl_threshold, bc_condition, self.is_gradient_clip)
            loss.update(nu_loss)
            mask = self.get_mask(bc_condition)
            loss['kl_div'] = kl_div[mask.to(torch.bool)].mean().detach().cpu().item()
            loss['kl_threshold'] = kl_threshold[mask.to(torch.bool)].mean().cpu().item()
        # update the target network
        if t % self.target_net_update_freq == 0:
            self.update_target_model(self.target_qf1, self.qf1)
            self.update_target_model(self.target_qf2, self.qf2)

        return loss
