import os
import torch

import scipy as sp
import cupy as cp

from tqdm import tqdm
from annoy import AnnoyIndex
from pprint import PrettyPrinter
#from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.metrics.pairwise import euclidean_distances
from cuml.neighbors import kneighbors_graph
from scipy.sparse.csgraph import dijkstra
from sklearn.preprocessing import MinMaxScaler

from cuml import KMeans
from cuml.metrics.cluster.silhouette_score import cython_silhouette_score

import matplotlib.pyplot as plt


def normalize_features(_x, _y):
    x_scale = MinMaxScaler()
    y_scale = MinMaxScaler()
    _new_x = x_scale.fit_transform(_x)
    _new_y = y_scale.fit_transform(_y)
    return _new_x, _new_y, x_scale, y_scale


def normalize_samples(_x, _scaler):
    _new_x = _scaler.transform(_x)
    return _new_x


def load_spaces(_ds_name, _kg_name, _lm_name, _ns, _reverse, _ptss):
    if _ptss:
        kge_name = 'triples_{}-{}-ht-{}_300_{}.pt'.format(_ds_name,
                                                          _kg_name,
                                                          _ns, _ns)
    else:
        kge_name = '{}_{}_concat_triples.pt'.format(_ds_name, _kg_name)
    ste_name = '{}_{}_space.pt'.format(_ds_name,
                                       _lm_name)
    wd = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    dd = os.path.join(wd, 'data')
    kgp = os.path.join(dd, _ds_name)
    print(kgp)
    triples_path = os.path.join(kgp, 'triples')
    sentence_path = os.path.join(kgp, 'sentences')
    print(triples_path)
    print(sentence_path)

    _kg_space = torch.load(os.path.join(triples_path, kge_name))
    print(_kg_space.shape)
    _se_space = torch.load(os.path.join(sentence_path, ste_name))
    print(_se_space.shape)
    return _kg_space, _se_space


def precompute_cost_matrices(_xs, _ys, _type, _plot):
    """
    A function to compute cost matrices for two given data domains.

    :param _xs: The representations of the first space, pytorch matrix.
    :param _ys: The representations of the second space, pytorch matrix.
    :param _type: Type of computation: euclidean distance (slow) or ball-tree (fast).
    :param _plot: Binary indicator to turn plotting on/off.
    :return: Two numpy cost matrices _c1, _c2 for each respective space.
    """
    if _type == 'euc':
        xs = _xs  # .detach().numpy()
        ys = _ys  # .detach().numpy()
        print('Computing cost for source...')
        _c1 = sp.spatial.distance.cdist(xs, xs, metric="cosine")
        #_c1 = euclidean_distances(xs, xs)
        print('... Done!')
        print('Computing cost for target...')
        _c2 = sp.spatial.distance.cdist(ys, ys, metric="cosine")
        print('... Done!')
        _c1 /= _c1.max()
        _c2 /= _c2.max()
        #ep = 1e-5
        #_c1 += ep
        #_c2 += ep
    elif _type == 'knn':
        xs = _xs  # .detach().numpy()
        ys = _ys  # .detach().numpy()
        nn_x = int(.2 * xs.shape[0])
        nn_y = int(.2 * ys.shape[0])
        #print('Computing cost for source...')
        _neig_x = kneighbors_graph(xs, nn_x, mode='distance').get()  #.todense()
        _c1 = dijkstra(_neig_x, directed=False, return_predecessors=False)
        _c1 /= _c1.max()
        #print('... Done!')
        #print('Computing cost for target...')
        _neig_y = kneighbors_graph(ys, nn_y, mode='distance').get()
        _c2 = dijkstra(_neig_y, directed=False, return_predecessors=False)
        _c2 /= _c2.max()
        #print('... Done!')
    else:
        print('Undefined cost function, exiting...')
        sys.exit()
    if _plot:
        plt.figure()
        plt.subplot(121)
        plt.imshow(_c1)
        plt.title('Triple Pairwise Distances')
        plt.subplot(122)
        plt.imshow(_c2)
        plt.title('Sentence Pairwise Distances')
        plt.show()
    return _c1, _c2


if __name__ == "__main__":
    load_spaces('nytfb', 'complex', 'gem', 5, True)
