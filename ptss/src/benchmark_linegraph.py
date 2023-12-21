import ast
import pickle
import json
import pandas as pd
import torch
import multiprocessing
import functools
import os
import numpy as np
import random

from tqdm import tqdm


def load_triple_map(_kg, _ns):
    wd = os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + os.pardir + os.sep + os.pardir)
    dp = "/kge/data/{}/"
    basepath = wd + dp.format(_kg) + 'triple-arrays.pkl'
    with open(basepath, 'rb') as f:
        _t2id = pickle.load(f)
    _lg_edges = pd.read_csv(wd + dp.format(_kg) + 'line_graph.csv', header=None)
    _lg_edges.columns = ['in', 'out']
    _lg_edges = _lg_edges.sample(frac=_ns, random_state=42, replace=False)
    try:
        _lg_ids = pd.read_csv(wd + dp.format(_kg) + 'triple_index_df.csv')
    except FileNotFoundError:
        triple_idx = []
        heads = []
        rels = []
        tails = []
        with open(wd + dp.format(_kg) + 'triples2ids.csv') as f:
            d = f.readlines()
        _ld_json = d[0].split('),')
        for j in _ld_json:
            idx = j.strip('{').strip(' ').split(':')
            triple_idx.append(idx[0])
            arr = idx[1].strip(' array([').strip(']').split(',')
            heads.append(int(arr[0].strip()))
            rels.append(int(arr[1].strip()))
            try:
                tails.append(int(arr[2].strip()))
            except ValueError:
                tails.append(int(arr[2].strip().strip('])}')))
        _lg_ids = pd.DataFrame({'triple': triple_idx, 'head': heads,
                                'rel': rels, 'tail': tails})
        _lg_ids.to_csv(wd + dp.format(_kg) + 'triple_index_df.csv', index=False)
    return _t2id, _lg_edges, _lg_ids


def fetch_embeddings(_path):
    mod = torch.load(_path, map_location=torch.device('cuda:0'))
    try:
        _ents = mod['model'][0]['_entity_embedder.embeddings.weight'].cpu()
        _rels = mod['model'][0]['_relation_embedder.embeddings.weight'].cpu()
    except KeyError:
        _ents = mod['model'][0]['_entity_embedder._embeddings.weight'].cpu()
        _rels = mod['model'][0]['_relation_embedder._embeddings.weight'].cpu()
    return _ents, _rels


def score_triples(_t1, _t2, _ent_embs, _rel_embs):
    h1 = _ent_embs[_t1[0]].unsqueeze(0)
    h2 = _ent_embs[_t2[0]].unsqueeze(0)
    r1 = _rel_embs[_t1[1]].unsqueeze(0)
    r2 = _rel_embs[_t2[1]].unsqueeze(0)
    t1 = _ent_embs[_t1[2]].unsqueeze(0)
    t2 = _ent_embs[_t2[2]].unsqueeze(0)
    h_sim = torch.cosine_similarity(h1, h2, ).detach().numpy()
    r_sim = torch.cosine_similarity(r1, r2).detach().numpy()
    t_sim = torch.cosine_similarity(t1, t2).detach().numpy()
    _score = np.mean([h_sim, r_sim, t_sim])
    return _score


def apply_scoring(_df, _t2idx):
    _pos_scores = []
    _neg_scores = []
    for idx, vals in tqdm(_df.iterrows()):
        t1 = _t2idx[vals['in']]
        t2 = _t2idx[vals['out']]
        t3 = _t2idx[vals['corr']]
        pos = score_triples(t1, t2, ents, rels)
        neg = score_triples(t1, t3, ents, rels)
        _pos_scores.append(pos)
        _neg_scores.append(neg)
    _df['pos_score'] = _pos_scores
    _df['neg_score'] = _neg_scores
    return _df

def score_wrapper(_df):
    return apply_scoring(_df, t2id)

def parallel_score(_triple_df, _t2idx, _ent_embs, _rel_embs):
    num_cores = multiprocessing.cpu_count() - 1
    num_partitions = num_cores
    df_split = np.array_split(_triple_df, num_partitions)
    pool = multiprocessing.Pool(num_cores)
    df = pd.concat(pool.map(score_wrapper, df_split))
    pool.close()
    pool.join()
    return df


def negative_edges(_df):
    corr = []
    for _, row in tqdm(_df.iterrows()):
        sub = lg_edges[lg_edges['in'] == row['in']]
        true_list = sub['out'].tolist()
        corr_list = [x for x in edge_ids if x not in true_list]
        corr.append(random.choice(corr_list))
    _df['corr'] = corr
    return _df


def parallel_negative(_triple_df, _all_edges):
    num_cores = multiprocessing.cpu_count() - 1
    num_partitions = num_cores
    df_split = np.array_split(_triple_df, num_partitions)
    pool = multiprocessing.Pool(num_cores)
    df = pd.concat(pool.map(negative_edges, df_split))
    pool.close()
    pool.join()
    return df


if __name__ == "__main__":
    kgs = ['fb15k-237']
    model_paths = ['pytorch-models/{}-complex.pt',
                   'pytorch-models/{}-conve.pt',
                   'pytorch-models/{}-distmult.pt',
                   'pytorch-models/{}-rescal.pt',
                   'pytorch-models/{}-rotate.pt',
                   'pytorch-models/{}-transe.pt'
                   ]
    for ns in [.1, .2, .3, .4, .5, .6, .7, .8, .9]:
        for kg in kgs:
            t2id, lg_edges, lg_ids = load_triple_map(kg, ns)
            all_edges = np.amax(lg_edges)
            print('Max edges: {}'.format(all_edges))
            edge_ids = [x for x in range(all_edges)]
            for m in model_paths:
                mn = m.split('/')[1].split('-')[1].split('.')[0]
                try:
                    df = pd.read_csv('ptss-benchmarks/{}/triple_scores_{}_{}_{}_emb_line.csv'.format(kg, kg, mn, ns))
                except FileNotFoundError:
                    try:
                        lgn_edges = pd.read_csv('ptss-benchmarks/{}/triple_scores_{}_{}_{}_neg_samp.csv'.format(kg, kg, mn, ns))
                    except FileNotFoundError:
                        lgn_edges = parallel_negative(lg_edges, edge_ids)
                        lgn_edges.to_csv('ptss-benchmarks/{}/triple_scores_{}_{}_{}_neg_samp.csv'.format(kg, kg, mn, ns), index=False)
                    ents, rels = fetch_embeddings(m.format(kg))
                    print(len(lgn_edges))
                    scores = parallel_score(lgn_edges, t2id, ents, rels)
                    scores.to_csv('ptss-benchmarks/{}/triple_scores_{}_{}_{}_emb_line.csv'.format(kg, kg, mn, ns),
                                  index=False)
                print('Finished triple_scores_{}_{}_{}_{}_emb_line'.format(kg, kg, mn, ns))
