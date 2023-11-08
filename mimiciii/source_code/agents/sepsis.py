from agents.DQN import *
from agents.SAC import *
import torch
import numpy as np
import torch.nn.functional as F

class DQN_regularization(DQN):
    def __init__(self, static_policy=False, env=None, config=None, log_dir='./logs'):
        super().__init__(static_policy, env, config, log_dir)

        # loss regularization term
        self.reg_lambda = config.REG_LAMBDA
        self.reward_threshold = config.REWARD_THRESHOLD

    # TODO: check this compute loss function is correct or not
    def compute_loss(self, batch_vars):
        '''
            loss function = E[Q_double-target - Q_estimate]^2 + lambda * max(|Q_estimate| - Q_threshold, 0)
            Q_double-target = reward + gamma * Q_double-target(next_state, argmax_a(Q(next_state, a)))
        '''
        states, actions, rewards, next_states, dones, indices, weights = batch_vars
        self.model.train()
        q_values = self.model(states).gather(1, actions)
        with torch.no_grad():
            max_next_action = self.get_max_next_state_action(next_states)
            target_q_values = self.target_model(next_states).gather(1, max_next_action)
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
    def __init__(self, static_policy=False, env=None, config=None, log_dir='./logs'):
        super().__init__(static_policy, env, config, log_dir)

    def append_to_replay(self, s, a, r, s_, a_, done, SOFA):
        self.memory.push((s, a, r, s_, a_, done, SOFA))

    def prep_minibatch(self):
        '''
        Returns:
            batch_state: expected shape (B, S)
            batch_action: expected shape (B, D)
            batch_reward: expected shape (B, 1)
            batch_next_state: expected shape (B, S)
            batch_done: expected shape (B, 1)
            batch_SOFA: expected shape (B, 1)
            indices: a list of index
            weights: expected shape (B,)
        '''
        # random transition batch is taken from replay memory
        transitions, indices, weights = self.memory.sample(self.batch_size)
        
        states, actions, rewards, next_states, next_actions, dones, SOFAs = zip(*transitions)

        states = torch.tensor(np.array(states), device=self.device, dtype=torch.float)
        actions = torch.tensor(np.array(actions), device=self.device, dtype=torch.int64).view(-1, 1)
        rewards = torch.tensor(np.array(rewards), device=self.device, dtype=torch.float).view(-1, 1)
        next_states = torch.tensor(np.array(next_states), device=self.device, dtype=torch.float)
        next_actions = torch.tensor(np.array(next_actions), device=self.device, dtype=torch.int64).view(-1, 1)
        dones = torch.tensor(np.array(dones), device=self.device, dtype=torch.float).view(-1, 1)
        SOFAs = torch.tensor(np.array(SOFAs), device=self.device, dtype=torch.float).view(-1, 1)

        # check shape
        assert states.dim() == 2 and states.shape[1] == self.num_feats
        assert actions.dim() == 2 and actions.shape[1] == 1
        assert rewards.dim() == 2 and rewards.shape[1] == 1
        assert next_states.dim() == 2 and next_states.shape[1] == self.num_feats
        assert next_actions.dim() == 2 and next_actions.shape[1] == 1
        assert dones.dim() == 2 and dones.shape[1] == 1
        assert SOFAs.dim() == 2 and SOFAs.shape[1] == 1

        return states, actions, rewards, next_states, next_actions, dones, SOFAs, indices, weights

    # TODO: check this compute loss function is correct or not
    def compute_loss(self, batch_vars):
        states, actions, rewards, next_states, next_actions, dones, SOFAs, indices, weights = batch_vars
        self.model.train()
        q_values = self.model(states).gather(1, actions)
        next_q_values = self.model(next_states)
        max_next_actions = torch.argmax(next_q_values, dim=1, keepdim=True)
        with torch.no_grad():
            target_next_q_values = self.target_model(next_states)
        max_target_next_actions = torch.argmax(target_next_q_values, dim=1, keepdim=True)
        target_next_q_values_softmax = F.softmax(target_next_q_values, dim=1)
        sigma = target_next_q_values_softmax.gather(1, max_next_actions)
        phi = target_next_q_values_softmax.gather(1, max_target_next_actions)
        p = phi / (phi + sigma)
        target_q_values = p * target_next_q_values.max(dim=1)[0].view(-1, 1) + (1 - p) * target_next_q_values.gather(1, max_next_actions)
        # expertise
        expert_q_values = target_next_q_values.gather(1, next_actions)
        target_q_values = torch.where(SOFAs < 4, expert_q_values, target_q_values)

        if self.priority_replay:
            diff = (q_values - (rewards + self.gamma * target_q_values * (1 - dones)))
            self.memory.update_priorities(indices, diff.detach().squeeze().abs().cpu().numpy().tolist())
            loss = ((0.5 * diff.pow(2))* weights).mean()
        else:
            loss = F.mse_loss(q_values, rewards + self.gamma * target_q_values * (1 - dones))
        return loss


class SAC_BC_E(SAC):
    def __init__(self, static_policy=False, env=None, config=None, log_dir='./logs') -> None:
        super().__init__(static_policy, env, config, log_dir) 

    # TODO: override any function if neccesary, and fix update function
    def update(self, t):
        # ref: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_atari.py
        self.actor.train()
        self.qf1.train()
        self.qf2.train()

        states, actions, rewards, next_states, dones, SOFAs, indices, weights = self.prep_minibatch()

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