import scipy as sp
import cupy as cp

from tqdm import tqdm
from annoy import AnnoyIndex
from pprint import PrettyPrinter
#from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.metrics.pairwise import euclidean_distances
from cuml.neighbors import kneighbors_graph
from scipy.sparse.csgraph import dijkstra

from cuml import KMeans
from cuml.metrics.cluster.silhouette_score import cython_silhouette_score

import matplotlib.pyplot as plt


def compute_cost_matrices(_xs, _ys, _type, _plot):
    """
    A function to compute cost matrices for two given data domains.

    :param _xs: The representations of the first space, pytorch matrix.
    :param _ys: The representations of the second space, pytorch matrix.
    :param _type: Type of computation: euclidean distance (slow) or ball-tree (fast).
    :param _plot: Binary indicator to turn plotting on/off.
    :return: Two numpy cost matrices _c1, _c2 for each respective space.
    """
    if _type == 'euc':
        xs = _xs.detach().numpy()
        ys = _ys.detach().numpy()
        print('Computing cost for source...')
        _c1 = sp.spatial.distance.cdist(xs, xs, metric="euclidean")
        #_c1 = euclidean_distances(xs, xs)
        print('... Done!')
        print('Computing cost for target...')
        _c2 = sp.spatial.distance.cdist(ys, ys, metric="euclidean")
        print('... Done!')
        _c1 /= _c1.max()
        _c2 /= _c2.max()
        ep = 1e-3
        _c1 += ep
        _c2 += ep
    elif _type == 'knn':
        xs = _xs.detach().numpy()
        ys = _ys.detach().numpy()
        nn_x = int(.2 * xs.shape[0])
        nn_y = int(.2 * ys.shape[0])
        print('Computing cost for source...')
        _neig_x = kneighbors_graph(xs, nn_x, mode='distance').get()  #.todense()
        _c1 = dijkstra(_neig_x, directed=False, return_predecessors=False)
        _c1 /= _c1.max()
        print('... Done!')
        print('Computing cost for target...')
        _neig_y = kneighbors_graph(ys, nn_y, mode='distance').get()
        _c2 = dijkstra(_neig_y, directed=False, return_predecessors=False)
        _c2 /= _c2.max()
        print('... Done!')
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


def apply_clustering(_xs):
    nc = [5, 10, 15, 20, 25, 30, 353]
    scores = []
    for n in tqdm(nc):
        kmeans = KMeans(n_clusters=n, max_iter=300, init='scalable-k-means++')
        kmeans.fit(_xs)
        ss = cython_silhouette_score(_xs, kmeans.labels_, metric='euclidean')
        scores.append(ss)
    print(scores)