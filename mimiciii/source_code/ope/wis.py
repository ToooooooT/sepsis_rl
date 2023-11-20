import numpy as np
import pandas as pd


def WIS_estimator(action_probs: np.ndarray, expert_data: pd.DataFrame, id_index_map, clip_expected_return):
    '''
    Args:
        action_probs    : policy action probabilities; expected shape (B, D)
        expert_data     : original expert dataset (DataFrame)
        id_index_map    : indexes of each icustayid (dict)
    Returns:
        avg_policy_return: average policy return
        policy_return: expected return of each patient; numpy array expected shape (1, B)
    '''
    # compute all trajectory total reward and weight imporatance sampling
    gamma = 0.99
    num = len(id_index_map)
    policy_return = np.zeros((num,), dtype=np.float64) 
    weights = np.zeros((num, 50)) # assume the patient max length is 50 
    length = np.zeros((num,), dtype=np.int32) # the horizon length of each patient
    rhos = action_probs[np.arange(action_probs.shape[0]), expert_data.loc[:, 'action'].values]
    for i, id in enumerate(id_index_map.keys()):
        start, end = id_index_map[id][0], id_index_map[id][-1]
        assert(50 >= end - start + 1)
        reward = 0
        length[i] = int(end - start + 1)
        for j, index in enumerate(range(end, start - 1, -1)):
            # let the minimum probability be 0.01 to avoid nan
            weights[i, end - start - j] = max(rhos[index], 0.01)
            # total reward
            reward = gamma * reward + expert_data.loc[index, 'reward']

        policy_return[i] = np.cumprod(weights[i])[length[i] - 1] * reward

    for i, l in enumerate(length):
        w_H = np.cumprod(weights[l <= length], axis=1)[:, l - 1].mean()
        policy_return[i] /= w_H

    policy_return = np.clip(policy_return, -clip_expected_return, clip_expected_return)
    return policy_return.mean(), policy_return.reshape(1, -1)