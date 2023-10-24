import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import copy
import os
import random

# gamma = 0.99
device = 'cpu'


class DistributionalDQN(nn.Module):
    def __init__(self, state_dim, n_actions, ):
        super(DistributionalDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.fc_val = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.fc_adv = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, state):
        conv_out = self.conv(state)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + adv - adv.mean(dim=1, keepdim=True)


class Dist_DQN(object):
    def __init__(self,
                 log_dir,
                 state_dim=37,
                 num_actions=25,
                 device='cpu',
                 gamma=0.999,
                 tau=0.1,
                 lr=0.0001,
                 batch_size=128,
                 ):
        self.device = device
        self.Q = DistributionalDQN(state_dim, num_actions, ).to(device)
        self.Q_target = copy.deepcopy(self.Q)
        self.tau = tau
        self.gamma = gamma
        self.num_actions = num_actions
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=lr)
        self.log_dir = log_dir
        self.batch_size = batch_size

    def save(self):
        os.makedirs(self.log_dir, exist_ok=True)
        torch.save(self.Q.state_dict(), os.path.join(self.log_dir, 'model.dump'))
        torch.save(self.optimizer.state_dict(), os.path.join(self.log_dir, 'optim.dump'))
    

    def load(self):
        fname_model = os.path.join(self.log_dir, "model.dump")
        fname_optim = os.path.join(self.log_dir, "optim.dump")

        if os.path.isfile(fname_model):
            self.Q.load_state_dict(torch.load(fname_model))
        else:
            assert False

        if os.path.isfile(fname_optim):
            self.optimizer.load_state_dict(torch.load(fname_optim))
        else:
            assert False


    def train(self, batchs):
        (state, next_state, action, next_action, reward, done, bloc_num, SOFAS) = batchs
        batch_s = self.batch_size
        uids = np.unique(bloc_num)
        num_batch = uids.shape[0] // batch_s  # 分批次
        record_loss = []
        sum_q_loss = 0
        Batch = 0
        for batch_idx in range(num_batch + 1):  # MAX_Iteration
            # ===============两种方法：1.随机抽取；2.顺序抽取===========================
            # -----1.随机抽取-----------------
            batch_uids = np.random.choice(uids, batch_s)  #replace=False
            # -----2.顺序抽取-----------------
            # batch_uids = uids[batch_idx * batch_s: (batch_idx + 1) * batch_s]
            batch_user = np.isin(bloc_num, batch_uids)
            state_user = state[batch_user, :]
            next_state_user = next_state[batch_user, :]
            action_user = action[batch_user]
            next_action_user = next_action[batch_user]
            reward_user = reward[batch_user]
            done_user = done[batch_user]
            SOFAS_user = SOFAS[batch_user]
            batch = (state_user, next_state_user, action_user, next_action_user, reward_user, done_user, SOFAS_user)
            loss = self.compute_loss(batch)
            sum_q_loss += loss.item()
            self.optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            self.optimizer.step()  # 更新
            if Batch % 25 == 0:
                record_loss1 = sum_q_loss / (Batch + 1)
                record_loss.append(record_loss1)
            # if Batch % 100 == 0:
            #     self.polyak_target_update()
            Batch += 1
        self.polyak_target_update()
        return record_loss

    def polyak_target_update(self):
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(param.data)

    def compute_loss(self, batch):
        state, next_state, action, next_action, reward, done, SOFA = batch
        gamma = 0.99
        end_multiplier = 1 - done
        batch_size = state.shape[0]
        range_batch = torch.arange(batch_size).long().to(device)
        log_Q_dist_prediction = self.Q(state)
        # modify the type of next_action to be index
        log_Q_dist_prediction1 = log_Q_dist_prediction[range_batch, action.type(torch.int64)]
        q_eval4nex = self.Q(next_state)
        max_eval_next = torch.argmax(q_eval4nex, dim=1)
        with torch.no_grad():
            Q_dist_target = self.Q_target(next_state)
            Q_target = Q_dist_target.clone().detach()
        Q_dist_eval = Q_dist_target[range_batch, max_eval_next]
        max_target_next = torch.argmax(Q_dist_target, dim=1)
        Q_dist_tar = Q_dist_target[range_batch, max_target_next]
        Q_target_pro = F.softmax(Q_target)
        pro1 = Q_target_pro[range_batch, max_eval_next] # sigma
        pro2 = Q_target_pro[range_batch, max_target_next] # phi
        Q_dist_star = (pro1 / (pro1 + pro2)) * Q_dist_eval + (pro2 / (pro1 + pro2)) * Q_dist_tar
        # modify the type of next_action to be index
        log_Q_experience = Q_dist_target[range_batch, next_action.type(torch.int64)] # Q_clini
        Q_experi = torch.where(SOFA < 4, log_Q_experience, Q_dist_star)
        targetQ1 = reward + (gamma * Q_experi * end_multiplier)
        return nn.SmoothL1Loss()(targetQ1, log_Q_dist_prediction1)

    def get_action(self, state):
        with torch.no_grad():
            est_q_values, actions = self.Q(state).max(dim=1)
            actions = actions.view(-1, 1).detach().cpu().numpy() # (B, 1)
            est_q_values = est_q_values.view(-1, 1).detach().cpu().numpy() # (B, 1)
            # assume policy take the max action in probability of 0.99 and any other actions of 0.01 
            action_probs = np.full((actions.shape[0], self.num_actions), 0.01)
            action_probs[np.arange(actions.shape[0]), actions[:, 0]] = 0.99

        return actions, action_probs, est_q_values

    def train_no_expertise(self, batchs):
        '''
            imitate the code in train which is the origin code from WD3QNE,
            but without expertise.
        '''
        (states, next_states, actions, rewards, dones) = batchs
        batch_s = self.batch_size
        n = states.shape[0]
        num_batch = n // batch_s
        sum_q_loss = 0
        # random
        batch_idxs = list(range(n))
        random.shuffle(batch_idxs)
        for i in range(num_batch + 1):
            batch_idx = batch_idxs[i * batch_s : min((i + 1) * batch_s, n)]
            state = states[batch_idx]
            action = actions[batch_idx]
            next_state = next_states[batch_idx]
            reward = rewards[batch_idx]
            done = dones[batch_idx]
            batch = (state, next_state, action, reward, done)
            loss = self.compute_loss_no_expertise(batch)
            sum_q_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return sum_q_loss

    def compute_loss_no_expertise(self, batch):
        '''
            imitate the code in compute loss which is the origin code from WD3QNE,
            but without expertise.
        '''
        state, next_state, action, reward, done = batch
        gamma = 0.99
        end_multiplier = 1 - done
        batch_size = state.shape[0]
        range_batch = torch.arange(batch_size).long().to(device)
        log_Q_dist_prediction = self.Q(state)
        # modify the type of next_action to be index
        log_Q_dist_prediction1 = log_Q_dist_prediction[range_batch, action.type(torch.int64)]
        q_eval4nex = self.Q(next_state)
        max_eval_next = torch.argmax(q_eval4nex, dim=1)
        with torch.no_grad():
            Q_dist_target = self.Q_target(next_state)
            Q_target = Q_dist_target.clone().detach()
        Q_dist_eval = Q_dist_target[range_batch, max_eval_next]
        max_target_next = torch.argmax(Q_dist_target, dim=1)
        Q_dist_tar = Q_dist_target[range_batch, max_target_next]
        Q_target_pro = F.softmax(Q_target)
        pro1 = Q_target_pro[range_batch, max_eval_next] # sigma
        pro2 = Q_target_pro[range_batch, max_target_next] # phi
        Q_dist_star = (pro1 / (pro1 + pro2)) * Q_dist_eval + (pro2 / (pro1 + pro2)) * Q_dist_tar
        targetQ1 = reward + (gamma * Q_dist_star * end_multiplier)
        return nn.SmoothL1Loss()(targetQ1, log_Q_dist_prediction1)
