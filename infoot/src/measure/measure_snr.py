import torch
import os
import numpy as onp
import seaborn as sns

from scipy.special import erfc
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
from matplotlib import pyplot as plt

from jax import numpy as np
from jax import jit
from jax import random

KEY = random.PRNGKey(0)


@jit
def shuffle(key, Xa, Xb):
    ''' Randomly shuffle examples in Xa and Xb along the zeroth axis.
    Args:
        key: /random PRNGkey
        Xa: (P,N) first array to shuffle
        Xb: (P,N) second array to shuffle

    Returns:
        Xaperm: (P,N) shuffled copy of Xa
        Xbperm: (P,N) shuffled copy of Xb
    '''
    keya, keyb = random.split(key)
    perma = random.shuffle(keya, np.arange(len(Xa)))
    permb = random.shuffle(keyb, np.arange(len(Xb)))

    return Xa[perma], Xb[permb]


@jit
def mshot_err(X):
    ''' Performs an m-shot learning experiment on a pair of shuffled manifolds X=(Xa,Xb).
    Args:
        m: # training examples
        X: X=(Xa,Xb), a pair of (P,N) object manifolds, pre-shuffled along the zeroth axis.

    Returns:
        erra: m-shot learning error evaluated on manifold a
        errb: m-shot learning error evaluated on manifold b
    '''
    m = 1
    Xa, Xb = X
    xatrain, xatest = np.split(Xa, (m,))
    xa = xatrain.mean(0)
    xbtrain, xbtest = np.split(Xb, (m,))
    xb = xbtrain.mean(0)
    x = np.vstack([xa, xb])

    distsa = ((x[:, None] - xatest[None]) ** 2).sum(-1)
    ya = distsa.argmin(0)

    distsb = ((x[:, None] - xbtest[None]) ** 2).sum(-1)
    yb = distsb.argmin(0)

    erra = (ya != 0).mean()
    errb = (yb != 1).mean()

    return erra, errb


@jit
def mshot_err_fast(key, Xa, Xb, _shots, _trials):
    ''' Performs a quick heuristic m-shot learning experiment on a pair of manifolds X=(Xa,Xb),
    allowing overlap between training and test examples.

    Args:
        X: X=(Xa,Xb), a pair of (P,N) object manifolds, pre-shuffled along the zeroth axis.

    Returns:
        erra: m-shot learning error evaluated on manifold a
        errb: m-shot learning error evaluated on manifold b
    '''
    m = _shots
    n_avg = _trials
    P = Xa.shape[0]
    print(P)
    keya, keyb = random.split(key)
    idxs_a = random.randint(keya, (m, n_avg), 0, P)
    idxs_b = random.randint(keyb, (m, n_avg), 0, P)
    print(idxs_a)

    # Prototypes
    xabar = Xa[idxs_a].mean(0)
    xbbar = Xb[idxs_b].mean(0)

    # Distances to prototypes
    daa = ((Xa[:, None] - xabar[None]) ** 2).sum(-1)
    dab = ((Xa[:, None] - xbbar[None]) ** 2).sum(-1)
    dba = ((Xb[:, None] - xabar[None]) ** 2).sum(-1)
    dbb = ((Xb[:, None] - xbbar[None]) ** 2).sum(-1)
    ha = -daa + dab
    hb = -dbb + dba

    erra = (ha < 0).mean()
    errb = (hb < 0).mean()

    return erra, errb


def h(_x):
    return erfc(_x/np.sqrt(2))/2


def snr(_signal, _bias, _D, _overlap, _m):
    _snr = .5*(_signal + _bias/_m) / np.sqrt(1/_D/_m + _overlap*(1+1/_m) + 1/_D/_m**2)
    return _snr


def find_errors(_manifolds, _trials, _key, _shots):
    """

    :param _manifolds: List of manifolds for one shot comparisons.
    :param _trials: Number of instances to run.
    :return:
    """
    K = len(_manifolds)
    errs_a = []
    errs_std_a = []
    errs_b = []
    errs_std_b = []

    for a in range(K):
        Xa = np.array(_manifolds[a])
        for b in range(a + 1, K):
            Xb = np.array(_manifolds[b])
            erra = []
            errb = []
            for _ in range(_trials):
                key, _ = random.split(_key)
                # erratmp, errbtmp = mshot_err_fast(key, Xa, Xb, _shots, _trials)
                erratmp,errbtmp = mshot_err(shuffle(key, Xa, Xb))
                erra.append(erratmp)
                errb.append(errbtmp)
            errs_a.append(np.stack(erra).mean())
            errs_std_a.append(np.stack(erra).std())
            errs_b.append(np.stack(errb).mean())
            errs_std_b.append(np.stack(errb).std())
    return errs_a, errs_std_a, errs_b, errs_std_b


def compute_geometry(_manifolds):
    """
    Computes the geometry as in
    :param _manifolds: A list of manifolds.
    :return:
    """
    # Radius, centers, subspaces
    Rs = []
    centers = []
    Us = []
    P = _manifolds[0].shape[0]
    print('Seting P to {}'.format(P))
    for manifold in tqdm(_manifolds):
        center = manifold.mean(0)
        centers.append(center)
        _, R, U = np.linalg.svd(manifold - center)
        Rs.append(R[:P])
        Us.append(U[:P])
    Rs = np.stack(Rs)
    centers = np.stack(centers)
    Us = np.stack(Us)

    # Overlaps
    K = len(_manifolds)
    ss = []
    csa = []
    csb = []
    for a in tqdm(range(K)):
        for b in range(K):
            if a != b:
                # Center-subspace
                dx0 = centers[a] - centers[b]
                dx0hat = dx0 / np.linalg.norm(dx0)
                costheta_a = Us[a] @ dx0hat
                csa.append((costheta_a ** 2 * Rs[a] ** 2).sum() / (Rs[a] ** 2).sum())
                costheta_b = Us[b] @ dx0hat
                csb.append((costheta_b ** 2 * Rs[b] ** 2).sum() / (Rs[a] ** 2).sum())

                # Subspace-subspace
                cosphi = Us[a] @ Us[b].T
                ss_overlap = (cosphi ** 2 * Rs[a][:, None] ** 2 * Rs[b] ** 2).sum() / (Rs[a] ** 2).sum() ** 2
                ss.append(ss_overlap)
            else:
                csa.append(np.nan)
                csb.append(np.nan)
                ss.append(np.nan)
    csa = np.stack(csa).reshape(K, K)
    csb = np.stack(csb).reshape(K, K)
    ss = np.stack(ss).reshape(K, K)

    return Rs, Us, centers, csa, csb, ss


def few_shot_error_analysis(_source_mfld, _target_mnfld):
    """
    Computes geometries and few-shot errors for source and target manifolds.

    :param _source_mfld: Source manifold.
    :param _target_mnfld: Target manifold
    :return:
    """
    p_src, d_src = _source_mfld.shape
    p_trg, d_trg = _target_mnfld.shape
    manifolds = np.stack([_source_mfld, _target_mnfld])
    Rs, Us, centers, csa, csb, ss = compute_geometry(manifolds)

    # add noisy projection
    an = np.random.randn(d_src, p_src) / np.sqrt(p_src)
    bn = np.random.randn(d_trg, p_trg) / np.sqrt(p_trg)


def load_spaces(_ds_name, _kg_name, _lm_name, _ns, _reverse, _ptss):
    if _ptss:
        kge_name = 'triples_{}-{}-ht-{}_300_{}.pt'.format(_ds_name,
                                                          _kg_name,
                                                          _ns, _ns)
    else:
        kge_name = '{}_{}_concat_triples.pt'.format(_ds_name, _kg_name)
    ste_name = '{}_{}_space.pt'.format(_ds_name,
                                       _lm_name)
    wd = os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + os.pardir)
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


def compute_dsvd(_rs, _centers):
    dists = squareform(pdist(centers))
    dist_norm = dists / np.sqrt((Rs ** 2).mean(-1)[:, None])

    # Dsvds = np.sum(Rs ** 2, axis=-1) ** 2 / np.sum(Rs ** 4, axis=-1)
    Dsvds = np.sum(Rs ** 2) ** 2 / np.sum(Rs ** 4)
    print('Mean Dsvd: ' + str(np.mean(Dsvds)))
    return Dsvds, dist_norm


def snr_matrix(_shot, _size, _center, _dist_norm, _rs, _csa, _csb, _nn, _dsvds):
    m = _shot
    P = _size
    K = _center.shape[0]
    bias = (_rs ** 2).sum(-1) / (_rs ** 2).sum(-1)[:, None] - 1
    signal = _dist_norm ** 2 + bias / m
    noise = _csa + _csb / m + _nn / m
    noise += 1 / _dsvds / m
    noise += 1 / _dsvds / m ** 2 / 2 * (1 - _dsvds / P / m)
    noise += ((_rs ** 2).sum(-1) / (_rs ** 2).sum(-1)[:, None]) / _dsvds[:, None] / m ** 2 / 2 * (1 - _dsvds / P / m)
    noise = np.sqrt(noise)
    snr = 1 / 2 * signal / noise
    return snr


def error_matrix(_manifolds):
    # Compute error
    m = 1
    n_avg = 10
    # something wrong with this loop, manifolds need to be reshaped
    p_src, d_src = _manifolds[0].shape
    p_trg, d_trg = _manifolds[1].shape
    _src2 = []
    _trg2 = []
    for k in range(n_avg):
        an = onp.random.randn(p_src, d_src) / np.sqrt(p_src)
        bn = onp.random.randn(p_trg, d_trg) / np.sqrt(p_trg)
        _src2.append(_manifolds[0]*an)
        _trg2.append(_manifolds[1]*bn)
    _src2 = np.stack(_src2).reshape(d_src, p_src, n_avg)
    _trg2 = np.stack(_trg2).reshape(d_trg, p_trg, n_avg)
    print('Batched manifolds with shape {}'.format(_src2.shape))
    err_all = onp.zeros((p_src, p_trg))
    for a in tqdm(range(p_src)):
        Xa = _src2[a]
        for b in range(p_trg):
            Xb = _trg2[b]
            errs = []
            for _ in range(n_avg):
                perma = onp.random.permutation(len(Xa))
                permb = onp.random.permutation(len(Xb))
                xa, ya = onp.split(Xa[perma], (m,))
                xb, yb = onp.split(Xb[permb], (m,))
                w = (xa - xb).mean(0)
                mu = (xa + xb).mean(0) / 2
                # pred = w.dot(xa.T) - (w*mu).sum(-1, keepdims=True)
                h = ya @ w - w @ mu
                err = (h < 0).mean()
                # err = h.mean()
                errs.append(err)
            err_all[a, b] = onp.mean(errs)
    onp.fill_diagonal(err_all, onp.nan)
    return err_all


if __name__ == "__main__":
    src = 'rescal'
    trg = 'glove'
    _plot = True
    kg, sen = load_spaces('nytfb', src, trg, '5', False, True)
    for bs in [256, 512, 1024, 2048]:
        manis = [kg[0:bs].cpu().detach().numpy(),
                 sen[0:bs].cpu().detach().numpy()]
        manis2 = [kg[0:bs].cpu().detach().numpy().T,
                 sen[0:bs].cpu().detach().numpy().T]
        Rs, Us, centers, csa, csb, ss = compute_geometry(manis)
        dsvd, dnorm = compute_dsvd(Rs, centers)
        ea, ea_st, eb, eb_st = find_errors(manis, 50, KEY, 1)
        errs_full = np.triu(squareform(ea)) + np.tril(squareform(eb))
        print('one-shot error estimate: {}'.format(1 - errs_full.mean()))
        err_all = error_matrix(manis)
        print(err_all)
        if _plot:
            with sns.axes_style('ticks'):
                plt.imshow(err_all)
                plt.colorbar()
                plt.savefig('{}-{}-{}-fewshot-ptss.png'.format(src, trg, bs))
        print(1 - np.nanmean(err_all))
        # snr = snr_matrix(1, bs, centers, dnorm, Rs, , dsvd)
        # print(snr)
