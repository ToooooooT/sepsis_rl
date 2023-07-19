import numpy as np

def WIS_estimator(actions, action_probs, expert_data, id_index_map, args):
    '''
    Args:
    actions         : policy action (tensor)
    expert_data     : original expert dataset (DataFrame)
    id_index_map    : indexes of each icustayid (dict)
    Returns:
        avg_policy_return: average policy return
        avg_expert_return: average expert return
        policy_return: expected return of each patient; expected shape (B,)
    '''
    # compute all trajectory total reward and weight imporatance sampling
    gamma = 0.99
    num = len(id_index_map)
    policy_return = np.zeros((num,), dtype=np.float64) 
    expert_return = np.zeros((num,)) 
    weights = np.zeros((num, 50))
    length = np.zeros((num,), dtype=np.int32) # the horizon length of each patient
    for i, id in enumerate(id_index_map.keys()):
        start, end = id_index_map[id][0], id_index_map[id][-1]
        assert(50 >= end - start + 1)
        reward = 0
        length[i] = int(end - start + 1)
        for j, index in enumerate(range(end, start - 1, -1)):
            # assume policy take the max action in probability of 0.99 and any othe actions of 0.01 
            if args.agent == 'D3QN':
                weights[i, end - start - j] = 0.99 if int(actions[index]) == int(expert_data.loc[index, 'action']) else 0.01
            elif args.agent == 'SAC':
                # let the minimum probability be 0.01 to avoid nan
                weights[i, end - start - j] = max(action_probs[index, int(expert_data.loc[index, 'action'])], 0.01)
            # total reward
            reward = gamma * reward + expert_data.loc[index, 'reward']

        policy_return[i] = np.cumprod(weights[i])[length[i] - 1] * reward
        expert_return[i] = reward 

    for i, l in enumerate(length):
        w_H = np.cumprod(weights[l <= length], axis=1)[:, l - 1].mean()
        policy_return[i] /= w_H

    policy_return = np.clip(policy_return, -25, 25)
    return policy_return.mean(), expert_return.mean(), policy_return