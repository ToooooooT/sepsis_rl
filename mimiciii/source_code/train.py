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
import uuid
import shutil
from tqdm import tqdm
import torch.nn as nn
from collections import OrderedDict
import json
import hashlib

from agents import DQN_regularization, WDQNE, SAC_BC_E, SAC_BC, SAC, BaseAgent, CQL, CQL_BC, CQL_BC_E
from utils import (Config, plot_action_dist, animation_action_distribution,
                    plot_pos_neg_action_dist, plot_diff_action_SOFA_dist,
                    plot_diff_action, plot_survival_rate, plot_expected_return_distribution,
                    plot_action_diff_survival_rate)
from network import DuellingMLP
from ope import WIS, DoublyRobust, FQE, QEstimator, PHWIS, PHWDR, BaseEstimator

pd.options.mode.chained_assignment = None

def hash_string(input: str):
    md5_hash = hashlib.md5()
    md5_hash.update(input.encode('utf-8'))
    return md5_hash.hexdigest()

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset_version", type=str, help="dataset version", default='MEAN')
    parser.add_argument("--reward_type", type=int, help="reward function type", default=0)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=128)
    parser.add_argument("--fqe_batch_size", type=int, help="batch_size", default=256)
    parser.add_argument("--episode", type=int, help="episode", default=120000)
    parser.add_argument("--fqe_episode", type=int, help="episode", default=150)
    parser.add_argument("--use_pri", type=int, help="use priority replay", default=0)
    parser.add_argument("--q_lr", type=float, help="q function learning rate", default=3e-4)
    parser.add_argument("--pi_lr", type=float, help="policy learning rate", default=3e-4)
    parser.add_argument("--fqe_lr", type=float, help="fitted q function learning rate", default=1e-4)
    parser.add_argument("--reg_lambda", type=int, help="regularization term coeficient", default=5)
    parser.add_argument("--use_state_augmentation", action="store_true", help="use state augmentaion")
    parser.add_argument("--state_augmentation_type", type=str, help="state augmentation type (Gaussian, Uniform, Mixup, Adversarial)", default='Gaussian')
    parser.add_argument("--state_augmentation_num", type=int, help="number of state augmentation", default=2)
    parser.add_argument("--is_sofa_threshold_below", action="store_true", help="employ behavior cloning when sofa score smaller then sofa threshold")
    parser.add_argument("--use_sofa_cv", action="store_true", help="use sofa cv as condition for behavior cloning")
    parser.add_argument("--sofa_threshold", type=float, help="sofa threshold with behavior cloning", default=5)
    parser.add_argument("--kl_threshold_exp", type=float, help="exponential term of the kl threshold exponential method", default=1.5)
    parser.add_argument("--kl_threshold_coef", type=float, help="coefficient term of the kl threshold exponential method", default=0.15)
    parser.add_argument("--bc_kl_beta", type=float, help="regularization term coeficient", default=2e-1)
    parser.add_argument("--correlation", type=str, help="correlation between indicator and BC", default="inverse")
    parser.add_argument("--agent", type=str, help="agent type", default="D3QN")
    parser.add_argument("--phy_epsilon", type=float, help="physician action distribution probabilities", default=0.005)
    parser.add_argument("--bc_type", type=str, help="behavior cloning type", default="cross_entropy")
    parser.add_argument("--kl_threshold_type", type=str, help="type of method to compute kl threshold", default="step")
    parser.add_argument("--use_pi_b_est", action="store_true", help="use estimate behavior policy action probabilities for OPE")
    parser.add_argument("--use_pi_b_kl", action="store_true", help="use estimate behavior policy action probabilities for KL in BC")
    parser.add_argument("--clip_expected_return", type=float, help="the value of clipping expected return", default=np.inf)
    parser.add_argument("--test_dataset", type=str, help="test dataset", default="test")
    parser.add_argument("--valid_freq", type=int, help="validation frequency", default=3000)
    parser.add_argument("--env_model_path", type=str, help="path of environment model", default="env_model.pth")
    parser.add_argument("--clf_model_path", type=str, help="path of classifier model", default="LG_clf.sav")
    parser.add_argument("--cpu", action="store_true", help="use cpu")
    parser.add_argument("--gradient_clip", action="store_true", help="gradient clipping in range (-1, 1)")
    parser.add_argument("--seed", type=int, help="random seed", default=10)
    parser.add_argument("--num_worker", type=int, help="number of worker to handle data loader", default=8)
    parser.add_argument("--load_checkpoint", action="store_true", help="load checkpoint")
    args = parser.parse_args()
    return args

def get_agent(args, log_path: str, env_spec: dict, config: Config):
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

def add_dataset_to_replay(train_data: dict, agent: DQN_regularization):
    # put all transitions in replay buffer
    s = train_data['s']
    a = train_data['a']
    r = train_data['r']
    s_ = train_data['s_']
    a_ = train_data['a_']
    done = train_data['done']
    sofa = train_data['SOFA']
    sofa_cv = train_data['SOFA_CV'] if args.dataset_version == 'MEAN' else sofa

    if isinstance(agent, SAC_BC_E) or isinstance(agent, CQL_BC_E):
        data = list(zip(*[s, a, r, s_, done, sofa, sofa_cv]))
        agent.memory.read_data(data)
    elif isinstance(agent, WDQNE):
        data = list(zip(*[s, a, r, s_, a_, done, sofa]))
        agent.memory.read_data(data)
    elif (isinstance(agent, DQN_regularization) or isinstance(agent, SAC_BC) 
            or isinstance(agent, SAC) or isinstance(agent, CQL_BC) or isinstance(agent, CQL)):
        data = list(zip(*[s, a, r, s_, done]))
        agent.memory.read_data(data)
    else:
        raise NotImplementedError

def training(
    agent: DQN_regularization, 
    ope_estimators: OrderedDict[str, BaseEstimator], 
    valid_dict: dict, 
    config: Config, 
    args
):
    '''
    Args:
        train_data      : processed training dataset
        valid_dataset   : original valid dataset (DataFrame)
        valud_dict      : processed validation dataset
    '''
    max_expected_return = {k: -np.inf for k in ope_estimators.keys()}
    hists = [] # save model actions of validation in every episode 
    VALID_FREQ = args.valid_freq
    LOG_FREQ = VALID_FREQ // 10
    valid_data = valid_dict['data']
    
    if args.load_checkpoint:
        start = agent.load_checkpoint()
    else:
        start = 0

    for i in tqdm(range(start + 1, config.EPISODE + 1)):
        loss = agent.update(i)
        log_metrics = loss

        if i % VALID_FREQ == 0:
            result = f'[EPISODE {i}] |' # for showing output
            agent.save_checkpoint(i)
            actions, action_probs = testing(valid_data, agent)

            # estimate expected return
            for method, estimator in ope_estimators.items():
                expected_return, _ = estimator.estimate(
                    policy_action_probs=action_probs,
                    behavior_action_probs=valid_dict['pi_b'],
                    agent=agent,
                    q=ope_estimators['FQE'].Q
                )

                if expected_return > max_expected_return[method]:
                    max_expected_return[method] = expected_return
                    agent.save(name=method + '_model.pth')

                log_metrics[method] = expected_return

                result += f' {method}: {expected_return:.5f},'

            print(result)

            # store actions in histogram to show animation
            hists.append(np.bincount(actions.reshape(-1), minlength=25))

        if i % LOG_FREQ == 0:
            mlflow.log_metrics(log_metrics, i)

    animation_action_distribution(hists, agent.log_dir)


def testing(test_data: dict, agent: BaseAgent) -> tuple[np.ndarray, np.ndarray]:
    '''
    Returns:
        actions     : np.ndarray; expected shape (B, 1)
        action_probs: np.ndarray; expected shape (B, D)
    '''
    states = torch.tensor(
        test_data['s'], 
        device=agent.device, 
        dtype=torch.float
    ).view(-1, agent.num_feats)

    with torch.no_grad():
        actions, _, _, action_probs = agent.get_action_probs(states)

    return actions.cpu().numpy(), action_probs.cpu().numpy()


def evaluation(
    agent: BaseAgent,
    test_dict: dict,
    test_dataset: pd.DataFrame,
    estimator: BaseEstimator, 
    method: str, 
    q: nn.Module | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    print(f'Start {method} testing...')
    policy_actions, policy_action_probs = testing(test_dict['data'], agent)

    policy_action_col = f'policy action {method}'
    policy_iv_col = f'policy iv {method}'
    policy_vaso_col = f'policy vaso {method}'

    test_dataset[policy_action_col] = policy_actions.reshape(-1,)
    test_dataset[policy_iv_col] = (policy_actions // 5).reshape(-1,)
    test_dataset[policy_vaso_col] = (policy_actions % 5).reshape(-1,)

    avg_return, returns = estimator.estimate(
        policy_action_probs=policy_action_probs,
        behavior_action_probs=test_dict['pi_b'],
        agent=agent,
        q=q
    )

    # plot action distribution
    if method == 'FQE-final' or 'final' not in method:
        negative_traj = test_dataset.query('mortality_90d == 1.0')
        positive_traj = test_dataset.query('mortality_90d != 1.0')
        mlflow.log_figure(
            plot_action_dist(policy_actions, test_dataset),
            f'fig/{method}/test_action_distribution.png')
        mlflow.log_figure(
            plot_pos_neg_action_dist(positive_traj, negative_traj, policy_action_col),
            f'fig/{method}/pos_neg_action_compare.png'
        )
        mlflow.log_figure(
            plot_diff_action_SOFA_dist(positive_traj, negative_traj, policy_action_col),
            f'fig/{method}/diff_action_SOFA_dist.png'
        )
        pos_fig, neg_fig = plot_diff_action(positive_traj, negative_traj, policy_action_col)
        mlflow.log_figure(pos_fig, f'fig/{method}/pos_diff_action_compare.png')
        mlflow.log_figure(neg_fig, f'fig/{method}/neg_diff_action_compare.png')
        low_fig, medium_fig, high_fig = plot_action_diff_survival_rate(
            train_dataset, 
            test_dataset,
            policy_iv_col,
            policy_vaso_col
        )
        mlflow.log_figure(low_fig, f'fig/{method}/diff_action_mortality_low_SOFA.png')
        mlflow.log_figure(medium_fig, f'fig/{method}/diff_action_mortality_medium_SOFA.png')
        mlflow.log_figure(high_fig, f'fig/{method}/diff_action_mortality_high_SOFA.png')

    return avg_return, returns


if __name__ == '__main__':
    args = parse_args()

    # generate random path for temporary storing before move to mlflow-artifacts
    path = hash_string(json.dumps(vars(args)) + str(uuid.uuid4()))

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
    dataset_path = f"../data/final_dataset/{args.dataset_version}/reward_type={args.reward_type}"

    # train
    train_dataset = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
    with open(os.path.join(dataset_path, 'train.pkl'), 'rb') as file:
        train_dict = pickle.load(file)
    train_data = train_dict['data']

    # validation
    valid_dataset = pd.read_csv(os.path.join(dataset_path, 'valid.csv'))
    with open(os.path.join(dataset_path, 'valid.pkl'), 'rb') as file:
        valid_dict = pickle.load(file)

    # test
    test_dataset = pd.read_csv(os.path.join(dataset_path, 'test.csv'))
    with open(os.path.join(dataset_path, 'test.pkl'), 'rb') as file:
        test_dict = pickle.load(file)

    # validation and test behavior policy probabilities
    valid_dict['pi_b'] = np.load(os.path.join(dataset_path, 'behavior_policy_prob_valid.npy'))
    test_dict['pi_b'] = np.load(os.path.join(dataset_path, 'behavior_policy_prob_test.npy'))

    ######################################################################################
    # Hyperparameters
    ######################################################################################
    config = Config()

    if args.cpu:
        config.DEVICE = torch.device("cpu")
    else:
        config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device : {config.DEVICE}')

    config.EPISODE = int(args.episode)
    config.USE_PRIORITY_REPLAY = args.use_pri
    config.Q_LR = args.q_lr
    config.PI_LR = args.pi_lr
    config.BATCH_SIZE = args.batch_size
    config.USE_PRIORITY_REPLAY = args.use_pri
    config.EXP_REPLAY_SIZE = len(train_data['s'])
    config.IS_GRADIENT_CLIP = args.gradient_clip
    config.REG_LAMBDA = args.reg_lambda
    config.BC_KL_BETA = args.bc_kl_beta
    config.BC_TYPE = args.bc_type
    config.PHY_EPSILON = args.phy_epsilon
    config.SOFA_THRESHOLD = args.sofa_threshold
    config.USE_SOFA_CV = args.use_sofa_cv
    config.IS_SOFA_THRESHOLD_BELOW = args.is_sofa_threshold_below
    config.KL_THRESHOLD_TYPE = args.kl_threshold_type
    config.KL_THRESHOLD_EXP = args.kl_threshold_exp
    config.KL_THRESHOLD_COEF = args.kl_threshold_coef
    config.CORRELATION = args.correlation
    config.USE_PI_B_EST = args.use_pi_b_est
    config.USE_PI_B_KL = args.use_pi_b_kl
    config.USE_STATE_AUGMENTATION = args.use_state_augmentation
    config.STATE_AUGMENTATION_TYPE = args.state_augmentation_type
    config.STATE_AUGMENTATION_NUM = args.state_augmentation_num

    ENV_SPEC = {'num_feats': train_data['s'].shape[1], 'num_actions': 25}

    LOG_PATH = os.path.join(f'./logs/{args.agent}', path)

    agent = get_agent(args, LOG_PATH, ENV_SPEC, config)

    os.makedirs(LOG_PATH)

    print('Adding dataset to replay buffer...')
    add_dataset_to_replay(train_data, agent)

    # start mlflow
    mlflow.set_tracking_uri("http://127.0.0.1:8787")
    experiment = mlflow.set_experiment(f"dataset_version={args.dataset_version}-reward_type={args.reward_type}-phy_epsilon-05")
    # start run
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=f"{args.agent}") as run:
        print(f'run id: {run.info.run_id}')
        params = config.get_hyperparameters()
        params['SEED'] = args.seed
        mlflow.log_params(params)

        # intialize estimators
        # * FQE method need to estimate first, since DR based method need the Q value from FQE
        ope_estimators = OrderedDict([
            ('FQE', FQE(
                agent, 
                train_dict['data'], 
                valid_dict['data'], 
                config, 
                args,
                Q=DuellingMLP(agent.num_feats, agent.num_actions, hidden_size=agent.hidden_size),
                target_Q=DuellingMLP(agent.num_feats, agent.num_actions, hidden_size=agent.hidden_size)
            )), 
            ('WIS', WIS(agent, valid_dict['data'], config, args)), 
            ('PHWIS', PHWIS(agent, valid_dict['data'], config, args)), 
            ('DR', DoublyRobust(agent, valid_dict['data'], config, args)), 
            ('PHWDR', PHWDR(agent, valid_dict['data'], config, args)), 
            # ('QE', QEstimator(agent, valid_dict['data'], config, args))
        ])

        # Training
        print('Start training...')
        training(agent, ope_estimators, valid_dict, config, args)
        ope_estimators['FQE'].records.to_csv(os.path.join(agent.log_dir, 'fqe_loss.csv'), index=False)

        # Off-policy Policy Evaluation for last checkpoint
        df_ope_returns = pd.DataFrame()
        agent.load_checkpoint()
        for method, estimator in ope_estimators.items():
            estimator.reset_data(test_dict['data']) 
            method += '-final'
            avg_return, returns = evaluation(
                agent, 
                test_dict, 
                test_dataset, 
                estimator, 
                method,
                ope_estimators['FQE'].Q
            )
            df_ope_returns[method] = returns

        # Off-policy Policy Evaluation for best model (based on expected return) of each estimators
        for method, estimator in ope_estimators.items():
            agent.load(name=method + '_model.pth')
            if 'DR' in method:
                # * must train FQE's q
                ope_estimators['FQE'].train()
            avg_return, returns = evaluation(
                agent, 
                test_dict, 
                test_dataset, 
                estimator, 
                method,
                ope_estimators['FQE'].Q
            )
            df_ope_returns[method] = returns

        # plot expected return result
        df_ope_returns.to_csv(os.path.join(agent.log_dir, 'expected_returns.csv'), index=False)
        mlflow.log_figure(
            plot_expected_return_distribution(df_ope_returns), 
            'fig/expected_return_distribution.png'
        )
        fig, survival_rate_means, survival_rate_stds = plot_survival_rate(
            df_ope_returns,
            test_dict['id_index_map'], 
            test_dataset, 
        )
        mlflow.log_figure(fig, 'fig/survival_rate.png')

        # store result in text file
        result = 'expected returns:\n'
        for i, method in enumerate(df_ope_returns.columns):
            result += f'\t{method}: {df_ope_returns[method].mean():.3f}\n'
        result += 'survival rates:\n'
        for i, method in enumerate(df_ope_returns.columns):
            result += f'\t{method}: {survival_rate_means[i]:.3f} ({survival_rate_stds[i]:.3f})\n'

        mlflow.log_text(result, 'text/expected_return.txt')
        print(result)

        test_dataset.to_csv(os.path.join(agent.log_dir, 'test_data_predict.csv'), index=False)
        mlflow.log_artifacts(agent.log_dir, "states")

    shutil.rmtree(agent.log_dir)
