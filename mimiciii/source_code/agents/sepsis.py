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
        transitions, indices, weights = self.memory.sample(self.batch_size)
        
        states, actions, rewards, next_states, next_actions, dones, SOFAs = zip(*transitions)

        states = torch.tensor(np.array(states), device=self.device, dtype=torch.float)
        actions = torch.tensor(np.array(actions), device=self.device, dtype=torch.int64).view(-1, 1)
        rewards = torch.tensor(np.array(rewards), device=self.device, dtype=torch.float).view(-1, 1)
        next_states = torch.tensor(np.array(next_states), device=self.device, dtype=torch.float)
        next_actions = torch.tensor(np.array(next_actions), device=self.device, dtype=torch.int64).view(-1, 1)
        dones = torch.tensor(np.array(dones), device=self.device, dtype=torch.float).view(-1, 1)
        SOFAs = torch.tensor(np.array(SOFAs), device=self.device, dtype=torch.float).view(-1, 1)

        return states, actions, rewards, next_states, next_actions, dones, SOFAs, indices, weights

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

    def append_to_replay(self, s, a, r, s_, done, SOFA):
        self.memory.push((s, a, r, s_, done, SOFA))

    def prep_minibatch(self):
        transitions, indices, weights = self.memory.sample(self.batch_size)
        
        states, actions, rewards, next_states, dones, SOFAs = zip(*transitions)

        states = torch.tensor(np.array(states), device=self.device, dtype=torch.float)
        actions = torch.tensor(np.array(actions), device=self.device, dtype=torch.int64).view(-1, 1)
        rewards = torch.tensor(np.array(rewards), device=self.device, dtype=torch.float).view(-1, 1)
        next_states = torch.tensor(np.array(next_states), device=self.device, dtype=torch.float)
        dones = torch.tensor(np.array(dones), device=self.device, dtype=torch.float).view(-1, 1)
        SOFAs = torch.tensor(np.array(SOFAs), device=self.device, dtype=torch.float).view(-1, 1)

        return states, actions, rewards, next_states, dones, SOFAs, indices, weights

    def compute_actor_loss(self, states, actions, SOFAs):
        # TODO: use the SOFAs to do expertise
        _, _, log_pi, action_probs = self.get_action_probs(states)
        with torch.no_grad():
            qf1_values = self.qf1(states)
            qf2_values = self.qf2(states)
            min_qf_values = torch.min(qf1_values, qf2_values)
        # no need for reparameterization, the expectation can be calculated for discrete actions
        actor_loss = (action_probs * (self.alpha * log_pi - min_qf_values)).mean()
        bc_loss = F.cross_entropy(action_probs, actions.view(-1))
        # TODO: use a suitable coefficient of normalization term
        coef = self.actor_lambda / (action_probs * (self.alpha * log_pi - min_qf_values)).abs().mean().detach()
        total_loss = actor_loss * coef + bc_loss
        return total_loss, actor_loss * coef, bc_loss, action_probs, log_pi

    def update(self, t):
        self.actor.train()
        self.qf1.train()
        self.qf2.train()
        states, actions, rewards, next_states, dones, SOFAs, indices, weights = self.prep_minibatch()
        # update critic 
        qf_loss = self.compute_critic_loss(states, actions, rewards, next_states, dones, indices, weights)
        self.q_optimizer.zero_grad()
        qf_loss.backward()
        if self.is_gradient_clip:
            self.gradient_clip_q()
        self.q_optimizer.step()
        # update actor 
        total_loss, actor_loss, bc_loss, action_probs, log_pi = self.compute_actor_loss(states, actions, SOFAs)
        self.actor_optimizer.zero_grad()
        total_loss.backward()
        if self.is_gradient_clip:
            self.gradient_clip_actor()
        self.actor_optimizer.step()

        if self.autotune:
            # Entropy regularization coefficient training
            # reuse action probabilities for temperature loss
            alpha_loss = (action_probs.detach() * (-self.log_alpha.exp() * (log_pi + self.target_entropy).detach())).mean()
            self.a_optimizer.zero_grad()
            alpha_loss.backward()
            self.a_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
        # update the target network
        if t % self.target_net_update_freq == 0:
            self.update_target_model(self.target_qf1, self.qf1)
            self.update_target_model(self.target_qf2, self.qf2)

        return {'qf_loss': qf_loss.detach().cpu().item(), 'actor_loss': actor_loss.detach().cpu().item(), 'bc_loss': bc_loss.detach().cpu().item(), 'alpha_loss': alpha_loss.detach().cpu().item()}