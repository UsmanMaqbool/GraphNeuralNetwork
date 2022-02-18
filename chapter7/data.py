import them
import os.path as osp
import pickle
import numpy as np
import itertools
import scipy.sparse as sp
import urllib
from collections import namedtuple


Data = namedtuple('Data', ['x', 'y', 'adjacency_dict',
                           'train_mask', 'val_mask', 'test_mask'])


class CoraData(object):
    filenames = ["ind.cora.{}".format(name) for name in
                 ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']]

    def __init__(self, data_root="../data/cora", rebuild=False):
        """Cora data, including data download, processing, loading and other functions
        When a cache file for the data exists, the cache file will be used, otherwise it will be downloaded, processed, and cached to disk

        The processed data can be obtained through the property .data, which will return a data object, including the following parts:
            * x: the characteristics of the node, the dimension is 2708 * 1433, the type is np.ndarray
            * y: The label of the node, including a total of 7 categories, the type is np.ndarray
            * adjacency_dict: adjacency information, the type is dict
            * train_mask: The training set mask vector, the dimension is 2708, when the node belongs to the training set, the corresponding position is True, otherwise False
            * val_mask: Validation set mask vector, the dimension is 2708, when the node belongs to the validation set, the corresponding position is True, otherwise False
            * test_mask: The test set mask vector, the dimension is 2708, when the node belongs to the test set, the corresponding position is True, otherwise False

        Args:
        -------
            data_root: string, optional
                The directory where the data is stored, the original data path: ../data/cora
                Cache data path: {data_root}/ch7_cached.pkl
            rebuild: boolean, optional
                Whether the dataset needs to be rebuilt, when set to True, the data will also be rebuilt if there is cached data

        """
        self.data_root = data_root
        save_file = osp.join(self.data_root, "ch7_cached.pkl")
        if osp.exists(save_file) and not rebuild:
            print("Using Cached file: {}".format(save_file))
            self._data = pickle.load(open(save_file, "rb"))
        else:
            self._data = self.process_data()
            with open(save_file, "wb") as f:
                pickle.dump(self.data, f)
            print("Cached file: {}".format(save_file))

    @property
    def data(self):
        """Return Data object, including x, y, adjacency, train_mask, val_mask, test_mask"""
        return self._data

    def process_data(self):
        """
        Process data to get node features and labels, adjacency matrix, training set, validation set and test set
        Quoted from: https://github.com/rusty1s/pytorch_geometric
        """
        print("Process data ...")
        _, tx, allx, y, ty, ally, graph, test_index = [self.read_data(
            osp.join(self.data_root, name)) for name in self.filenames]
        train_index = np.arange(y.shape[0])
        val_index = np.arange(y.shape[0], y.shape[0] + 500)
        sorted_test_index = sorted(test_index)

        x = np.concatenate((allx, tx), axis=0)
        y = np.concatenate((ally, ty), axis=0).argmax(axis=1)

        x[test_index] = x[sorted_test_index]
        y[test_index] = y[sorted_test_index]
        num_nodes = x.shape[0]

        train_mask = np.zeros(num_nodes, dtype=np.bool)
        val_mask = np.zeros(num_nodes, dtype=np.bool)
        test_mask = np.zeros(num_nodes, dtype=np.bool)
        train_mask[train_index] = True
        val_mask[val_index] = True
        test_mask[test_index] = True
        adjacency_dict = graph
        print("Node's feature shape: ", x.shape)
        print("Node's label shape: ", y.shape)
        print("Adjacency's shape: ", len(adjacency_dict))
        print("Number of training nodes: ", train_mask.sum())
        print("Number of validation nodes: ", val_mask.sum())
        print("Number of test nodes: ", test_mask.sum())

        return Data(x=x, y=y, adjacency_dict=adjacency_dict,
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    @staticmethod
    def build_adjacency(adj_dict):
        """Create an adjacency matrix from the adjacency list"""
        edge_index = []
        num_nodes = len(adj_dict)
        for src, dst in adj_dict.items():
            edge_index.extend([src, v] for v in dst)
            edge_index.extend([v, src] for v in dst)
        # remove duplicate edges
        edge_index = list(k for k, _ in itertools.groupby(sorted(edge_index)))
        edge_index = np.asarray(edge_index)
        adjacency = sp.coo_matrix((np.ones(len(edge_index)),
                                   (edge_index[:, 0], edge_index[:, 1])),
                                  shape=(num_nodes, num_nodes), dtype="float32")
        return adjacency

    @staticmethod
    def read_data(path):
        """Use a different way to read raw data for further processing"""
        name = osp.basename(path)
        if name == "ind.cora.test.index":
            out = np.genfromtxt(path, dtype="int64")
            return out
        else:
            out = pickle.load(open(path, "rb"), encoding="latin1")
            out = out.toarray() if hasattr(out, "toarray") else out
            return out