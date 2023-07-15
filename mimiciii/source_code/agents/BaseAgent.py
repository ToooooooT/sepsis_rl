import os
import torch


class BaseAgent(object):
    def __init__(self, config, env, log_dir='./log', agent_dir='./saved_agents'):
        self.model = None
        self.target_model = None
        self.optimizer = None

        self.log_dir = log_dir # log directory
        self.agent_dir = agent_dir # saved agents directory

        self.action_selections = [0 for _ in range(env['num_actions'])] # the frequency of each action be selected

        self.device = config.device


    def save(self):
        if not os.path.exists(self.agent_dir):
            os.mkdir(self.agent_dir)
        torch.save(self.model.state_dict(), os.path.join(self.log_dir, 'model.dump'))
        torch.save(self.optimizer.state_dict(), os.path.join(self.log_dir, 'optim.dump'))
    

    def load(self):
        fname_model = os.path.join(self.log_dir, "model.dump")
        fname_optim = os.path.join(self.log_dir, "optim.dump")

        if os.path.isfile(fname_model):
            self.model.load_state_dict(torch.load(fname_model))
            self.target_model.load_state_dict(self.model.state_dict())
        else:
            assert False

        if os.path.isfile(fname_optim):
            self.optimizer.load_state_dict(torch.load(fname_optim))
        else:
            assert False


    def save_action(self, actions):
        '''
        Args:
            actions: numpy.array; expected shape (B, 1)
        '''
        # save the frequency of each action be selected
        n = actions.shape[0]
        self.action_selections = [0 for _ in range(len(self.action_selections))]
        for action in actions:
            self.action_selections[int(action)] += (1.0 / n)