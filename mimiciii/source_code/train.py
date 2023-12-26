######################################################################################
# Import Package
######################################################################################
import numpy as np
import pandas as pd
import torch
import os
import random
import pickle
from argparse import ArgumentParser
import mlflow

from agents import DQN_regularization, WDQNE, SAC_BC_E, SAC_BC, SAC, BaseAgent, CQL, CQL_BC, CQL_BC_E
from utils import Config, plot_action_dist, plot_estimate_value, \
                animation_action_distribution, plot_pos_neg_action_dist, plot_diff_action_SOFA_dist, \
                plot_diff_action, plot_survival_rate, plot_expected_return_distribution, \
                plot_action_diff_survival_rate
from network import DuellingMLP
from ope import WIS, DoublyRobust, FQE, QEstimator

pd.options.mode.chained_assignment = None

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--reward_type", type=int, help="reward function type", default=0)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=128)
    parser.add_argument("--fqe_batch_size", type=int, help="batch_size", default=256)
    parser.add_argument("--episode", type=int, help="episode", default=150000)
    parser.add_argument("--fqe_episode", type=int, help="episode", default=150)
    parser.add_argument("--use_pri", type=int, help="use priority replay", default=1)
    parser.add_argument("--q_lr", type=float, help="q function learning rate", default=3e-4)
    parser.add_argument("--pi_lr", type=float, help="policy learning rate", default=3e-4)
    parser.add_argument("--fqe_lr", type=float, help="fitted q function learning rate", default=1e-4)
    parser.add_argument("--reg_lambda", type=int, help="regularization term coeficient", default=5)
    parser.add_argument("--agent", type=str, help="agent type", default="D3QN")
    parser.add_argument("--bc_type", type=str, help="behavior cloning type", default="cross_entropy")
    parser.add_argument("--clip_expected_return", type=float, help="the value of clipping expected return", default=np.inf)
    parser.add_argument("--test_dataset", type=str, help="test dataset", default="test")
    parser.add_argument("--valid_freq", type=int, help="validation frequency", default=1000)
    parser.add_argument("--gif_freq", type=int, help="frequency of making validation action distribution gif", default=1000)
    parser.add_argument("--env_model_path", type=str, help="path of environment model", default="env_model.pth")
    parser.add_argument("--clf_model_path", type=str, help="path of classifier model", default="LG_clf.sav")
    parser.add_argument("--cpu", action="store_true", help="use cpu")
    parser.add_argument("--gradient_clip", action="store_true", help="gradient clipping in range (-1, 1)")
    parser.add_argument("--seed", type=int, help="random seed", default=10)
    parser.add_argument("--num_worker", type=int, help="number of worker to handle data loader", default=8)
    parser.add_argument("--load_checkpoint", action="store_true", help="load checkpoint")
    args = parser.parse_args()
    return args

def get_agent(args, log_path, env_spec, config):
    if args.agent == 'D3QN':
        agent = DQN_regularization(log_dir=log_path, env=env_spec, config=config)
    elif args.agent == 'WD3QNE':
        agent = WDQNE(log_dir=log_path, env=env_spec, config=config)
    elif args.agent == 'SAC':
        agent = SAC(log_dir=log_path, env=env_spec, config=config)
    elif args.agent == 'SAC_BC':
        agent = SAC_BC(log_dir=log_path, env=env_spec, config=config)
    elif args.agent == 'SAC_BC_E':
        agent = SAC_BC_E(log_dir=log_path, env=env_spec, config=config)
    elif args.agent == 'CQL':
        agent = CQL(log_dir=log_path, env=env_spec, config=config)
    elif args.agent == 'CQL_BC':
        agent = CQL_BC(log_dir=log_path, env=env_spec, config=config)
    elif args.agent == 'CQL_BC_E':
        agent = CQL_BC_E(log_dir=log_path, env=env_spec, config=config)
    else:
        raise NotImplementedError
    return agent

def add_dataset_to_replay(train_data, agent: DQN_regularization):
    # put all transitions in replay buffer
    s = train_data['s']
    a = train_data['a']
    r = train_data['r']
    s_ = train_data['s_']
    a_ = train_data['a_']
    done = train_data['done']
    SOFA = train_data['SOFA']

    if isinstance(agent, SAC_BC_E) or isinstance(agent, CQL_BC_E):
        data = [s, a, r, s_, done, SOFA]
        agent.memory.read_data(data)
    elif isinstance(agent, WDQNE):
        data = [s, a, r, s_, a_, done, SOFA]
        agent.memory.read_data(data)
    elif isinstance(agent, DQN_regularization) or isinstance(agent, SAC_BC) or isinstance(agent, SAC) or isinstance(agent, CQL_BC) or isinstance(agent, CQL):
        data = [s, a, r, s_, done]
        agent.memory.read_data(data)
    else:
        raise NotImplementedError

def training(agent: DQN_regularization, valid_dataset: pd.DataFrame, valid_dict: dict, config: Config, args):
    '''
    Args:
        train_data      : processed training dataset
        valid_dataset   : original valid dataset (DataFrame)
        valud_dict      : processed validation dataset
    '''
    wis_returns = []
    dr_returns = []
    fqe_returns = []
    qe_returns = []
    hists = [] # save model actions of validation in every episode 
    valid_freq = args.valid_freq
    gif_freq = args.gif_freq
    max_expected_return = -np.inf
    valid_data = valid_dict['data']
    dr = DoublyRobust(agent, valid_dict['data'], config, args, valid_dataset)
    wis = WIS(agent, valid_dict['data'], config, args)
    fqe = FQE(agent, 
              train_dict['data'], 
              valid_dict['data'], 
              config, 
              args,
              Q=DuellingMLP(agent.num_feats, agent.num_actions, hidden_size=agent.hidden_size),
              target_Q=DuellingMLP(agent.num_feats, agent.num_actions, hidden_size=agent.hidden_size))
    qe = QEstimator(agent, valid_dict['data'], config, args)

    if args.load_checkpoint:
        start = agent.load_checkpoint()
    else:
        start = 0

    for i in range(start + 1, config.EPISODE + 1):
        loss = agent.update(i)
        log_data = loss

        if i % valid_freq == 0:
            actions, action_probs = testing(valid_data, agent)
            # store actions in histogram to show animation
            if i % gif_freq == 0:
                hists.append(np.bincount(actions.reshape(-1), minlength=25))
            # estimate expected return
            wis_return, _ = wis.estimate(policy_action_probs=action_probs)
            wis_returns.append(wis_return)
            fqe_return, _  = fqe.estimate(agent=agent)
            fqe_returns.append(fqe_return)
            dr_return, _, _ = dr.estimate(policy_actions=actions, 
                                          policy_action_probs=action_probs, 
                                          agent=agent, 
                                          q=fqe.Q)
            dr_returns.append(dr_return)
            qe_return, _ = qe.estimate(agent=agent)
            qe_returns.append(qe_return)
            # currently use fqe to choose model
            if fqe_return > max_expected_return:
                max_expected_return = fqe_return
                agent.save()

            print(f'[EPISODE {i}] | WIS: {wis_return:.5f}, DR: {dr_return:.5f}, FQE : {fqe_return:.5f}, QE: {qe_return:.5f}')

            log_data["WIS"] = wis_return
            log_data["DR"] = dr_return
            log_data["FQE"] = fqe_return
            log_data["QE"] = qe_return

        if i % (valid_freq // 10) == 0:
            mlflow.log_metrics(log_data, i)

        agent.save_checkpoint(i)

    animation_action_distribution(hists, agent.log_dir)
    wis_returns = np.array(wis_returns)
    dr_returns = np.array(dr_returns)
    fqe_returns = np.array(fqe_returns)
    qe_returns = np.array(qe_returns)
    fig = plot_estimate_value(np.vstack((wis_returns, dr_returns, fqe_returns, qe_returns)), 
                        ['WIS', 'DR', 'FQE', 'QE'], 
                        valid_freq)
    mlflow.log_figure(fig, 'fig/valid estimate value.png')
    mlflow.log_table(fqe.records, 'table/fqe_loss.json')


def testing(test_dict: dict, agent: BaseAgent):
    '''
    Returns:
        actions     : np.ndarray; expected shape (B, 1)
        action_probs: np.ndarray; expected shape (B, D)
    '''
    states = torch.tensor(test_dict['s'], device=agent.device, dtype=torch.float).view(-1, agent.num_feats)

    with torch.no_grad():
        actions, _, _, action_probs = agent.get_action_probs(states)
        actions = actions.cpu().numpy()
        action_probs = action_probs.cpu().numpy()

    return actions, action_probs


if __name__ == '__main__':
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ######################################################################################
    # Load Dataset
    ######################################################################################
    '''
    train / valid / test dataset are original unnomalized dataset, with action and reward
    train / valid / test data contain (s, a, r, s_, done, SOFA, is_alive) transitions, with normalization
    '''
    dataset_path = "../data/final_dataset/"

    # train
    train_dataset = pd.read_csv(os.path.join(dataset_path, f'train_{args.reward_type}.csv'))
    icustayids = train_dataset['icustayid'].unique()
    with open(os.path.join(dataset_path, f'train_{args.reward_type}.pkl'), 'rb') as file:
        train_dict = pickle.load(file)
    train_data = train_dict['data']

    # validation
    valid_dataset = pd.read_csv(os.path.join(dataset_path, f'valid_{args.reward_type}.csv'))
    with open(os.path.join(dataset_path, f'valid_{args.reward_type}.pkl'), 'rb') as file:
        valid_dict = pickle.load(file)

    # test
    test_dataset = pd.read_csv(os.path.join(dataset_path, f'test_{args.reward_type}.csv'))
    with open(os.path.join(dataset_path, f'test_{args.reward_type}.pkl'), 'rb') as file:
        test_dict = pickle.load(file)
    test_data, test_id_index_map = test_dict['data'], test_dict['id_index_map']

    ######################################################################################
    # Hyperparameters
    ######################################################################################
    config = Config()

    if args.cpu:
        config.DEVICE = torch.device("cpu")
    else:
        config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config.EPISODE = int(args.episode)
    config.USE_PRIORITY_REPLAY = args.use_pri
    config.Q_LR = args.q_lr
    config.PI_LR = args.pi_lr
    config.BATCH_SIZE = args.batch_size
    config.USE_PRIORITY_REPLAY = args.use_pri
    config.EXP_REPLAY_SIZE = len(train_data['s'])
    config.IS_GRADIENT_CLIP = args.gradient_clip
    config.REG_LAMBDA = args.reg_lambda
    config.BC_TYPE = args.bc_type

    env_spec = {'num_feats': 49, 'num_actions': 25}

    path = f'{args.agent}/reward_type={args.reward_type}-test_episode={config.EPISODE}-batch_size={config.BATCH_SIZE}-use_pri={config.USE_PRIORITY_REPLAY}-q_lr={config.Q_LR}-pi_lr={config.PI_LR}-hidden_size={config.HIDDEN_SIZE}'
    if args.agent == 'D3QN':
        path += f'-reg_lambda={config.REG_LAMBDA}'
    if args.agent == 'SAC_BC_E' or args.agent == 'CQL_BC_E' or args.agent == 'SAC_BC' or args.agent == 'CQL_BC':
        path += f'-bc_type={config.BC_TYPE}'
    log_path = os.path.join('./logs', path)

    agent = get_agent(args, log_path, env_spec, config)

    os.makedirs(log_path, exist_ok=True)

    print('Adding dataset to replay buffer...')
    add_dataset_to_replay(train_data, agent)

    # start mlflow
    mlflow.set_tracking_uri("http://127.0.0.1:8787")
    experiment = mlflow.set_experiment(f"reward_type={args.reward_type}")
    # start run
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=f"{args.agent}") as run:
        print(f'run id: {run.info.run_id}')
        mlflow.log_params(config.get_hyperparameters())
        # Training
        print('Start training...')
        training(agent, valid_dataset, valid_dict, config, args)
        # Testing
        agent.load()

        print('Start testing...')
        policy_actions, policy_action_probs = testing(test_data, agent)

        test_dataset['policy action'] = policy_actions
        test_dataset['policy iv'] = policy_actions / 5
        test_dataset['policy vaso'] = policy_actions % 5

        # Off-policy Evaluation
        wis = WIS(agent, test_dict['data'], config, args)
        avg_wis_return, wis_returns = wis.estimate(policy_action_probs=policy_action_probs)
        fqe = FQE(agent, 
                train_dict['data'], 
                test_dict['data'], 
                config, 
                args,
                Q=DuellingMLP(agent.num_feats, agent.num_actions, hidden_size=agent.hidden_size),
                target_Q=DuellingMLP(agent.num_feats, agent.num_actions, hidden_size=agent.hidden_size))
        avg_fqe_return, fqe_returns = fqe.estimate(agent=agent)
        dr = DoublyRobust(agent, test_dict['data'], config, args, test_dataset)
        avg_dr_return, dr_returns, est_alive = dr.estimate(policy_action_probs=policy_action_probs, 
                                                           policy_actions=policy_actions, 
                                                           agent=agent,
                                                           q=fqe.Q)
        qe = QEstimator(agent, test_dict['data'], config, args)
        avg_qe_return, qe_returns = qe.estimate(agent=agent)

        # plot expected return result
        policy_returns = np.vstack((wis_returns, dr_returns, fqe_returns, qe_returns))
        mlflow.log_figure(plot_expected_return_distribution(policy_returns, ['WIS', 'DR', 'FQE', 'QE']), 
                          'fig/expected_return_distribution.png')
        mlflow.log_figure(plot_survival_rate(policy_returns, test_id_index_map, test_dataset, ['WIS', 'DR', 'FQE', 'QE']), 
                          'fig/survival_rate.png')

        # plot action distribution
        negative_traj = test_dataset.query('mortality_90d == 1.0')
        positive_traj = test_dataset.query('mortality_90d != 1.0')
        mlflow.log_figure(plot_action_dist(policy_actions, test_dataset),
                          'fig/test_action_distribution.png')
        mlflow.log_figure(plot_pos_neg_action_dist(positive_traj, negative_traj),
                          'fig/pos_neg_action_compare.png')
        mlflow.log_figure(plot_diff_action_SOFA_dist(positive_traj, negative_traj),
                          'fig/diff_action_SOFA_dist.png')
        pos_fig, neg_fig = plot_diff_action(positive_traj, negative_traj)
        mlflow.log_figure(pos_fig, 'fig/pos_diff_action_compare.png')
        mlflow.log_figure(neg_fig, 'fig/neg_diff_action_compare.png')
        low_fig, medium_fig, high_fig = plot_action_diff_survival_rate(train_dataset, test_dataset)
        mlflow.log_figure(low_fig, 'fig/diff_action_mortality_low_SOFA.png')
        mlflow.log_figure(medium_fig, 'fig/diff_action_mortality_medium_SOFA.png')
        mlflow.log_figure(high_fig, 'fig/diff_action_mortality_high_SOFA.png')

        # store result in text file
        mlflow.log_text(f'''
                        WIS : {avg_wis_return:.5f}
                        DR : {avg_dr_return:.5f}
                        FQE : {avg_fqe_return:.5f}
                        QE : {avg_qe_return:.5f}
                        Logistic regression survival rate: {est_alive.mean():.5f}
                        ''', 'text/expected_return.txt')
        # print result
        print(f'WIS estimator: {avg_wis_return:.5f}')
        print(f'DR estimator: {avg_dr_return:.5f}')
        print(f'FQE estimator: {avg_fqe_return:.5f}')
        print(f'QE estimator: {avg_qe_return:.5f}')
        print(f'Logistic regression survival rate: {est_alive.mean():.5f}')

        mlflow.log_table(test_dataset, 'table/test_data_predict.json')
        mlflow.log_artifacts(agent.log_dir, "states")
