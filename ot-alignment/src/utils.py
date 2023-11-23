import numpy as np
import ot
import time
import torch
import matplotlib.pyplot as plt

from cuml import UMAP
from tqdm import tqdm


def plot_spaces(_dset, _src, _trg, _src_labs, _trg_labs, _seed, _src_name, _trg_name, _eps):
    src_mapper = UMAP(n_components=2, n_neighbors=100, random_state=_seed)
    trg_mapper = UMAP(n_components=2, n_neighbors=100, random_state=_seed)
    src_umap = src_mapper.fit_transform(_src)
    trg_umap = trg_mapper.fit_transform(_trg)
    plt.figure()
    plt.subplot(121)
    plt.scatter(src_umap[:, 0], src_umap[:, 1], c=_src_labs, s=0.5)
    plt.title('{}-{}-{}'.format(_src_name, _eps, _seed))
    plt.subplot(122)
    plt.scatter(trg_umap[:, 0], trg_umap[:, 1], c=_trg_labs, s=0.5)
    plt.title('{}-{}-{}'.format(_trg_name, _eps, _seed))
    plt.savefig('{}_outputs/{}-{}-{}-{}.png'.format(_dset, _src_name, _trg_name, _seed, _eps))


def small_batch_sampler(_data, _weight, _batch_sz):
    """
    Adapted from killianFatras to compute minibatches.

    :param _data:
    :param _weight:
    :param _batch_sz:
    :return:
    """
    _n = _data.shape[0]
    _batch_ids = np.random.randint(0, _n, _batch_sz)
    indicate = 1
    while indicate > 0:
        indicate = 0
        for i in range(_batch_sz):
            for j in range(i+1, _batch_sz):
                if _batch_ids[i] == _batch_ids[j]:
                    _batch_ids[j] = np.random.randint(0, _n)
                    indicate += 1
    _batch_weights = ot.unif(_batch_sz)
    _minibatch = _data[_batch_ids]
    return _minibatch, _batch_weights, _batch_ids


def batch_sampler(_data, _weight, _batch_sz):
    """

    :param _data:
    :param _weight:
    :param _batch_sz:
    :return:
    """
    _batch_ids = np.random.choice(_data.shape[0], _batch_sz, replace=False, p=_weight)
    _batch_weights = ot.unif(_batch_sz)
    return _data[_batch_ids], _batch_weights, _batch_ids


def incremental_barycentric_mapping(_src, _trg, _src_a, _trg_b, _sbs, _tbs, _num_batches):
    """

    :param _src: Source data.
    :param _trg: Target data.
    :param _src_a: Source distribution.
    :param _trg_b: Target distribution.
    :param _sbs: Source batch size.
    :param _tbs: Target batch size.
    :param _num_batches: Number of batched couplings.
    :return:
    """
    _src = _src.detach().cpu().numpy()
    _trg = _trg.detach().cpu().numpy()
    new_src = np.zeros(_src.shape)
    new_trg = np.zeros(_trg.shape)
    num_src = _src.shape[0]
    num_trg = _trg.shape[0]
    for _ in tqdm(range(_num_batches)):
        _sub_src, _sub_src_w, _a_ids = batch_sampler(_src, _src_a, _sbs)
        _sub_trg, _sub_trg_w, _b_ids = batch_sampler(_trg, _trg_b, _tbs)
        _sub_m = ot.dist(_sub_src, _sub_trg, "sqeuclidean").copy()
        G0 = ot.emd(_sub_src_w, _sub_trg_w, _sub_m)
        new_src[_a_ids] += G0.dot(_trg[_b_ids])
        new_trg[_b_ids] += G0.T.dot(_src[_a_ids])
    new_src = 1./_num_batches * num_src * new_src
    new_trg = 1./_num_batches * num_trg * new_trg
    return new_src, new_trg


def normalize_embeddings(_normalize_vecs, _src, _trg):
    if _normalize_vecs == 'whiten':
        print("Normalizing embeddings with {}".format(_normalize_vecs))
        _src, _trg = center_embeddings(_src, _trg)
        _src, _trg = _whiten_embeddings(_src, _trg)
    elif _normalize_vecs == 'mean':
        print("Normalizing embeddings with {}".format(_normalize_vecs))
        _src, _trg = scale_embeddings(_src, _trg)
    elif _normalize_vecs == 'both':
        print("Normalizing embeddings with {}".format(_normalize_vecs))
        # Artexte - robust self learning
        _src, _trg = scale_embeddings(_src, _trg)
        _src, _trg = center_embeddings(_src, _trg)
        _src, _trg = scale_embeddings(_src, _trg)
    else:
        print('Warning: no normalization performed')
    return _src, _trg


def center_embeddings(_src, _trg):
    # this should mean center each dimension
    src_mean = _src.mean(axis=0)
    _src.data = _src - src_mean
    trg_mean = _trg.mean(axis=0)
    _trg.data = _trg - trg_mean
    return _src, _trg


def scale_embeddings(_src, _trg):
    src_norm = _src.norm(p=2, dim=1, keepdim=True)
    _src.data = _src.div(src_norm)
    trg_norm = _trg.norm(p=2, dim=1, keepdim=True)
    _trg.data = _trg.div(trg_norm)
    return _src, _trg


def _whiten_embeddings(_src, _trg):
    src_cov = cov(_src.T)
    u_src, s_src, v_src = torch.svd(src_cov)
    w_src = (v_src.T / torch.sqrt(s_src)).T
    _src.data = _src.data @ w_src.T
    trg_cov = cov(_trg.T)
    u_trg, s_trg, v_trg = torch.svd(trg_cov)
    w_trg = (v_trg.T / torch.sqrt(s_trg)).T
    _trg.data = _trg.data @ w_trg.T
    return _src, _trg


def cov(m, rowvar=True, inplace=False):
    """
    Estimate a covariance matrix given data.
    """
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    if inplace:
        m -= torch.mean(m, dim=1, keepdim=True)
    else:
        m = m - torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()
