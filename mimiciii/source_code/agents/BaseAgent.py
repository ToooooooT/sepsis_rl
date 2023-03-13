import pickle
import os
import csv
import torch


class BaseAgent(object):
    def __init__(self, config, env, log_dir='./log'):
        self.model = None
        self.target_model = None
        self.optimizer = None

        self.log_dir = log_dir # log directory

        self.rewards = [] # save the rewards

        self.action_log_frequency = config.ACTION_SELECTION_COUNT_FREQUENCY # the frequency to save the action selections into csv file
        self.action_selections = [0 for _ in range(env['num_actions'])] # the frequency of each action be selected
        self.action_log = 0

        if os.path.exists(os.path.join(self.log_dir, 'action_log.csv')):
            os.remove(os.path.join(self.log_dir, 'action_log.csv'))
        if os.path.exists(os.path.join(self.log_dir, 'td.csv')):
            os.remove(os.path.join(self.log_dir, 'td.csv'))

    def save(self):
        torch.save(self.model.state_dict(), './saved_agents/model.dump')
        torch.save(self.optimizer.state_dict(), './saved_agents/optim.dump')
    
    def load(self):
        fname_model = "./saved_agents/model.dump"
        fname_optim = "./saved_agents/optim.dump"

        if os.path.isfile(fname_model):
            self.model.load_state_dict(torch.load(fname_model))
            self.target_model.load_state_dict(self.model.state_dict())

        if os.path.isfile(fname_optim):
            self.optimizer.load_state_dict(torch.load(fname_optim))

    def save_replay(self):
        pickle.dump(self.memory, open('./saved_agents/exp_replay_agent.dump', 'wb'))

    def load_replay(self):
        fname = './saved_agents/exp_replay_agent.dump'
        if os.path.isfile(fname):
            self.memory = pickle.load(open(fname, 'rb'))

    def save_reward(self, reward):
        self.rewards.append(reward)

    def save_action(self, actions):
        # save the frequency of each action be selected
        n = actions.shape[0]
        self.action_selections = [0 for _ in range(len(self.action_selections))]
        for action in actions:
            self.action_selections[int(action)] += (1.0 / n)

    def save_td(self, td, tstep):
        with open(os.path.join(self.log_dir, 'td.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow((tstep, td))

    '''
    def save_sigma_param_magnitudes(self, tstep):
        with torch.no_grad():
            sum_, count = 0.0, 0.0
            for name, param in self.model.named_parameters():
                if param.requires_grad and 'sigma' in name:
                    sum_+= torch.sum(param.abs()).item()
                    count += np.prod(param.shape)
            
            if count > 0:
                with open(os.path.join(self.log_dir, 'sig_param_mag.csv'), 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow((tstep, sum_/count))

    '''
