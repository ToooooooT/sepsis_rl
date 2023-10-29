from DQN import DQN, WDQN
import torch
import numpy as np

class DQN_regularization(DQN):
    def __init__(self, static_policy=False, env=None, config=None, log_dir='./logs'):
        super().__init__(static_policy, env, config, log_dir)

    def append_to_replay(self, s, a, r, s_, done, SOFA):
        self.memory.push((s, a, r, s_, done, SOFA))

    def prep_minibatch(self):
        '''
        Returns:
            batch_state: expected shape (B, S)
            batch_action: expected shape (B, D)
            batch_reward: expected shape (B, 1)
            batch_next_state: expected shape (B, S)
            batch_done: expected shape (B, 1)
            indices: a list of index
            weights: expected shape (B,)
        '''
        # random transition batch is taken from replay memory
        transitions, indices, weights = self.memory.sample(self.batch_size)
        
        batch_state, batch_action, batch_reward, batch_next_state, batch_done, _ = zip(*transitions)

        batch_state = torch.tensor(np.array(batch_state), device=self.device, dtype=torch.float)
        batch_action = torch.tensor(np.array(batch_action), device=self.device, dtype=torch.int64).view(-1, 1)
        batch_reward = torch.tensor(np.array(batch_reward), device=self.device, dtype=torch.float).view(-1, 1)
        batch_next_state = torch.tensor(np.array(batch_next_state), device=self.device, dtype=torch.float)
        batch_done = torch.tensor(np.array(batch_done), device=self.device, dtype=torch.float).view(-1, 1)
        
        return batch_state, batch_action, batch_reward, batch_next_state, batch_done, indices, weights

    def compute_loss(self, batch_vars):
        '''
            TODO: fix this function
            loss function = E[Q_double-target - Q_estimate]^2 + lambda * max(|Q_estimate| - Q_threshold, 0)
            Q_double-target = reward + gamma * Q_double-target(next_state, argmax_a(Q(next_state, a)))
            Q_threshold = 20
            when die reward -15 else +15
        '''
        self.model.train()

        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        # estimate
        q_values = self.model(batch_state)
        current_q_values = q_values.gather(1, batch_action) 
        
        # compute target value
        with torch.no_grad():
            # unnecessary to compute gradient and do back propogation
            max_next_q_values = torch.zeros(self.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)
            if not empty_next_state_values:
                max_next_action = self.get_max_next_state_action(non_final_next_states)
                max_next_q_values[non_final_mask] = self.target_model(non_final_next_states).gather(1, max_next_action)

            # empirical hack to make the Q values never exceed the threshold - helps learning
            if self.reg_lambda > 0:
                max_next_q_values[max_next_q_values > self.reward_threshold] = self.reward_threshold
                max_next_q_values[max_next_q_values < -self.reward_threshold] = -self.reward_threshold

            expected_q_values = batch_reward + ((self.gamma ** self.nsteps) * max_next_q_values)

        td_error = (expected_q_values - current_q_values).pow(2)
        if self.priority_replay:
            self.memory.update_priorities(indices, (td_error / 10).detach().squeeze().abs().cpu().numpy().tolist()) # ?
            loss = 0.5 * (td_error * weights).mean() + self.reg_lambda * max(current_q_values.abs().max() - self.reward_threshold, 0)
            # loss = 0.5 * (td_error * weights).mean()
        else:
            loss = 0.5 * td_error.mean() + self.reg_lambda * max(current_q_values.abs().max() - self.reward_threshold, 0)
            # loss = 0.5 * td_error.mean()
        return loss


class WDQNE(WDQN):
    def __init__(self, static_policy=False, env=None, config=None, log_dir='./logs'):
        super().__init__(static_policy, env, config, log_dir)