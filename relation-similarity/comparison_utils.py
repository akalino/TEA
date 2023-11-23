import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import linalg as la


def make_diagonal(_a, _val):
    new_rows = []
    for j in range(_a.shape[0]):
        row = list(_a[j])
        row.insert(j, _val)
        new_rows.append(row)
    return np.array(new_rows)


def normalize(_m):
    _m_max, _m_min = _m.max(), _m.min()
    _new_m = (_m - _m_min) / (_m_max - _m_min)
    return _new_m


def plot_heat(_m, _sh):
    _gr = sns.heatmap(_m)
    if _sh:
        plt.savefig()
    return _gr


def von_neumann_entropy(_m):
    """
    Computes the von Neumann entropy of a matrix using the trace method.

    :param _m: A square metric.
    :return: Float, the von Neumann entropy.
    """
    try:
        _m = _m.detach().numpy()
    except AttributeError:
        pass
    r = _m * (la.norm(_m) / la.logm(np.matrix([[2]])))
    s = -np.matrix.trace(r)
    return s


def von_neumann_eigen(_m):
    """
    Computes the von Neumann entropy of a matrix using the eigenvalue method.

    :param _m: A square metric.
    :return: Float, the von Neumann entropy.
    """
    try:
        _m = _m.detach().numpy()
    except AttributeError:
        pass
    evs = la.eigvals(_m)
    nz_evs = np.matrix(np.array([x for x in evs.tolist() if x]))
    log_evs = np.matrix(np.log2(nz_evs))
    s = -np.dot(nz_evs, log_evs.H)
    return s
