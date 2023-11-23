import numpy as np
import json
import torch
import os
import pandas as pd
import sys

from comparison_utils import make_diagonal, normalize, plot_heat, von_neumann_entropy, von_neumann_eigen
from models.baseModel import fetch_embeddings


def jaccard_similarity(a, b):
    a = set(a)
    b = set(b)
    j = float(len(a.intersection(b))) / len(a.union(b))
    return j


def read_predicate_map(_path):
    """
    Loads predicate to index map.

    :param _path: Path to text file.
    :return: Dictionary with indices as keys, string names as values.
    """
    _map = {}
    with open(_path) as f:
        for line in f:
            vs = line.split("\t")
            if len(vs) > 1:
                _map[vs[1].strip("\n")] = vs[0]
    return _map


def top_five_candidates(_m, _set):
    """
    Computes a list of the top five most similar predicates for all available predicates.

    :param _m: A predicate similarity matrix.
    :param _p_map: Mapping of predicate indices to human-readable strings.
    :return:
    """
    _rp = {}
    mp = os.path.join('data', _set, 'relation2id.txt')
    map = read_predicate_map(mp)
    for j in range(len(_m)):
        try:
            scores = _m[j, :].numpy()
        except AttributeError:
            scores = _m[j, :]
        sort_score = scores.argsort()
        cands = sort_score[::-1][1:6]
        query = map[str(j)]
        matches = []
        for k in cands:
            matches.append(map[str(k)])
        while len(matches) < 5:
            matches.append('null')
        _rp[query] = matches
    return _rp


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def plot_kl(_set, _sh):
    _m = np.loadtxt('result/kl_prob_{}.txt'.format(_set))
    _b = normalize(_m)
    _c = make_diagonal(_b, 1.0)
    s_ent = von_neumann_entropy(_c)
    s_eig = von_neumann_eigen(_c)
    hm = plot_heat(_c, _sh)
    rp = top_five_candidates(_c, _set)
    return s_ent, s_eig, hm, rp


def plot_freq(_set, _sh):
    _c = np.genfromtxt('result/rel_matrix_{}.csv'.format(_set),
                       delimiter=',')
    s_ent = von_neumann_entropy(_c)
    s_eig = von_neumann_eigen(_c)
    hm = plot_heat(_c, _sh)
    rp = top_five_candidates(_c, _set)
    return s_ent, s_eig, hm, rp


def plot_vecs(_set, _mn, _sh):
    mp = os.path.join('data', _set,
                      '{}-{}.pt'.format(_set, _mn))
    entity_pretrain, relation_pretrain = fetch_embeddings(mp)
    num_rel = int(relation_pretrain.shape[0] / 2)
    _rels = relation_pretrain[0:num_rel, :]
    _c = sim_matrix(_rels,
                    _rels)
    s_ent = von_neumann_entropy(_c)
    s_eig = von_neumann_eigen(_c)
    hm = plot_heat(_c, _sh)
    rp = top_five_candidates(_c, _set)
    return s_ent, s_eig, hm, rp


def compute_overlaps(_rdict):
    """
    Computes the top 5 candidate Jaccard index for each predicate across multiple models.

    :param _rdict: Dictionary of all top 5 candidate predicates.
    :return:
    """
    for _ds in ['fb15k-237', 'wnrr']:
        mod_res = []
        results = _rdict[_ds]
        models = list(results.keys())
        all_preds = list(results[models[0]].keys())
        model_pairs = [(a, b) for idx, a in enumerate(models) for b in models[idx + 1:]]
        for p in model_pairs:
            print('Computing overlap of pairs {}'.format(p))
            mod1 = results[p[0]]
            mod2 = results[p[1]]
            sims = []
            for r in all_preds:
                try:
                    l1 = mod1[r]
                    l2 = mod2[r]
                    sims.append(jaccard_similarity(l1, l2))
                except KeyError:
                    sims.append(0)
            avg = sum(sims) / len(sims)
            print('Average Jaccard score {}'.format(avg))


if __name__ == "__main__":
    show = False
    out = []
    all_ranks = {'fb15k-237': {}, 'wnrr': {}}
    s1, s2, hm1, rp1 = plot_kl('fb15k-237', show)
    out.append(['fb15k-237', 'kl-div', s1, s2])
    all_ranks['fb15k-237']['kl-div'] = rp1

    s1, s2, hm2, rp2 = plot_freq('fb15k-237', show)
    out.append(['fb15k-237', 'pf-ipf', s1, s2])
    all_ranks['fb15k-237']['pf-ipf'] = rp2

    s1, s2, hm3, rp3 = plot_vecs('fb15k-237', 'complex', show)
    out.append(['fb15k-237', 'complex-sim', s1, s2])
    all_ranks['fb15k-237']['complex-sim'] = rp3

    s1, s2, hm4, rp4 = plot_vecs('fb15k-237', 'conve', show)
    out.append(['fb15k-237', 'conve-sim', s1, s2])
    all_ranks['fb15k-237']['conve-sim'] = rp4

    s1, s2, hm5, rp5 = plot_vecs('fb15k-237', 'distmult', show)
    out.append(['fb15k-237', 'distmult-sim', s1, s2])
    all_ranks['fb15k-237']['distmult-sim'] = rp5

    s1, s2, hm6, rp6 = plot_vecs('fb15k-237', 'rescal', show)
    out.append(['fb15k-237', 'rescal-sim', s1, s2])
    all_ranks['fb15k-237']['rescal-sim'] = rp6

    s1, s2, hm7, rp7 = plot_vecs('fb15k-237', 'rotate', show)
    out.append(['fb15k-237', 'rotate-sim', s1, s2])
    all_ranks['fb15k-237']['rotate-sim'] = rp7

    s1, s2, hm8, rp8 = plot_vecs('fb15k-237', 'transe', show)
    out.append(['fb15k-237', 'transe-sim', s1, s2])
    all_ranks['fb15k-237']['transe-sim'] = rp8

    s1, s2, hm9, rp9 = plot_kl('wnrr', show)
    out.append(['wnrr', 'kl-div', s1, s2])
    all_ranks['wnrr']['kl-div'] = rp9

    s1, s2, hm10, rp10 = plot_freq('wnrr', show)
    out.append(['wnrr', 'pf-ipf', s1, s2])
    all_ranks['wnrr']['pf-ipf'] = rp10

    s1, s2, hm11, rp11 = plot_vecs('wnrr', 'complex', show)
    out.append(['wnrr', 'complex-sim', s1, s2])
    all_ranks['wnrr']['complex-sim'] = rp11

    s1, s2, hm12, rp12 = plot_vecs('wnrr', 'conve', show)
    out.append(['wnrr', 'conve-sim', s1, s2])
    all_ranks['wnrr']['conve-sim'] = rp12

    s1, s2, hm13, rp13 = plot_vecs('wnrr', 'distmult', show)
    out.append(['wnrr', 'distmult-sim', s1, s2])
    all_ranks['wnrr']['distmult-sim'] = rp13

    s1, s2, hm14, rp14 = plot_vecs('wnrr', 'rescal', show)
    out.append(['wnrr', 'rescal-sim', s1, s2])
    all_ranks['wnrr']['rescal-sim'] = rp14

    s1, s2, hm15, rp15 = plot_vecs('wnrr', 'rotate', show)
    out.append(['wnrr', 'rotate-sim', s1, s2])
    all_ranks['wnrr']['rotate-sim'] = rp15

    s1, s2, hm16, rp16 = plot_vecs('wnrr', 'transe', show)
    out.append(['wnrr', 'transe-sim', s1, s2])
    all_ranks['wnrr']['transe-sim'] = rp16

    df = pd.DataFrame(out, columns=['dataset', 'method', 'entropy-trace', 'entropy-eig'])
    print(df)
    df.to_csv('entropy-metrics.csv', index=False)

    compute_overlaps(all_ranks)
    with open('result/predicate_rankings.json', 'w') as f:
        json.dump(all_ranks, f, indent=4)

