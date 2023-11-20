import numpy as np

def q_value_estimator(est_q_values: np.ndarray, id_index_map: dict):
    '''
    Args:
        id_index_map    : indexes of each icustayid (dict)
        est_q_values    : np.ndarray; expected shape (B, 1)
    Returns:
        avg_policy_return: average policy return
        policy_return    : expected return of each patient using q value to estimate; expected shape (1, B)
    '''
    policy_returns = np.zeros((len(id_index_map),))
    for i, id in enumerate(id_index_map.keys()):
        start = id_index_map[id][0]
        policy_returns[i] = est_q_values[start, 0]
    return policy_returns.mean(), policy_returns.reshape(1, -1)