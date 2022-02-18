import numpy as np


def sampling(src_nodes, sample_num, neighbor_table):
    """Sample the specified number of neighbor nodes according to the source node, note that the sampling with replacement is used;
    When the number of neighbor nodes of a node is less than the number of samples, the sampling results show duplicate nodes
    
    Arguments:
        src_nodes {list, ndarray} -- list of source nodes
        sample_num {int} -- the number of nodes to sample
        neighbor_table {dict} -- the mapping table of nodes to their neighbors
    
    Returns:
        np.ndarray -- a list of sampled results
    """
    results = []
    for sid in src_nodes:
        # Sampling with replacement from the node's neighbors
        res = np.random.choice(neighbor_table[sid], size=(sample_num, ))
        results.append(res)
    return np.asarray(results).flatten()


def multihop_sampling(src_nodes, sample_nums, neighbor_table):
    """Multi-order sampling based on source node
    
    Arguments:
        src_nodes {list, np.ndarray} -- source node id
        sample_nums {list of int} -- the number of samples to be sampled at each stage
        neighbor_table {dict} -- a map of nodes to their neighbors
    
    Returns:
        [list of ndarray] -- the result of each stage of sampling
    """
    sampling_result = [src_nodes]
    for k, hopk_num in enumerate(sample_nums):
        hopk_result = sampling(sampling_result[k], hopk_num, neighbor_table)
        sampling_result.append(hopk_result)
    return sampling_result