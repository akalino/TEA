import os
import pandas as pd
import multiprocessing
import numpy as np
import torch

from tqdm import tqdm


def fetch_embeddings(_path):
    mod = torch.load(_path, map_location=torch.device('cuda:0'))
    try:
        _ents = mod['model'][0]['_entity_embedder.embeddings.weight'].cpu()
        _rels = mod['model'][0]['_relation_embedder.embeddings.weight'].cpu()
    except KeyError:
        _ents = mod['model'][0]['_entity_embedder._embeddings.weight'].cpu()
        _rels = mod['model'][0]['_relation_embedder._embeddings.weight'].cpu()
    return _ents, _rels


def get_triples(_ds, _n):
    _df = pd.read_csv('ptss-benchmarks/triple_benchmarks_{}_{}.csv'.format(_ds, _n))
    print('Found {} benchmark triples for {} -- {} samples'.format(len(_df), _ds, _n))
    return _df


def score_triples(_t1, _t2, _ent_embs, _rel_embs):
    h1 = _ent_embs[_t1[0]].unsqueeze(0)
    h2 = _ent_embs[_t2[0]].unsqueeze(0)
    r1 = _ent_embs[_t1[1]].unsqueeze(0)
    r2 = _ent_embs[_t2[1]].unsqueeze(0)
    t1 = _ent_embs[_t1[2]].unsqueeze(0)
    t2 = _ent_embs[_t2[2]].unsqueeze(0)
    h_sim = torch.cosine_similarity(h1, h2, ).detach().numpy()
    r_sim = torch.cosine_similarity(r1, r2).detach().numpy()
    t_sim = torch.cosine_similarity(t1, t2).detach().numpy()
    _score = np.mean([h_sim, r_sim, t_sim])
    return _score


def apply_scoring(_df, _ent_embs, _rel_embs):
    _all_scores = []
    for idx, vals in tqdm(_df.iterrows()):
        t1 = [vals['true_head'], vals['true_rel'], vals['true_tail']]
        t2 = [vals['head_idx'], vals['rel_idx'], vals['tail_idx']]
        sim = score_triples(t1, t2, _ent_embs, _rel_embs)
        _all_scores.append(sim)
    _df['sim_score'] = _all_scores
    return _df


def apply_row_scoring(_row, _ent_embs, _rel_embs):
    t1 = [_row['true_head'], _row['true_rel'], _row['true_tail']]
    t2 = [_row['head_idx'], _row['rel_idx'], _row['tail_idx']]
    _sim = score_triples(t1, t2, _ent_embs, _rel_embs)
    _row['sim_score'] = _sim


def parallel_score(_triple_df, _ent_embs, _rel_embs):
    num_cores = multiprocessing.cpu_count()-1
    num_partitions = num_cores
    df_split = np.array_split(_triple_df, num_partitions)
    pool = multiprocessing.Pool(num_cores)
    df = pd.concat(pool.map(partial(apply_row_scoring, _ent_embs, _rel_embs), df_split))
    pool.close()
    pool.join()
    return df


if __name__ == "__main__":
    ds = ['nytfb']
    model_paths = [#'pytorch-models/{}-complex.pt',
                   #'pytorch-models/{}-conve.pt',
                   'pytorch-models/{}-distmult.pt',
                   #'pytorch-models/{}-rescal.pt',
                   #'pytorch-models/{}-rotate.pt',
                   #'pytorch-models/{}-transe.pt'
                   ]
    samples = [5]
    for d in ds:
        for ns in samples:
            trip_df = get_triples(d, ns)
            for mt in model_paths:
                mn = mt.split('/')[1].split('-')[1].split('.')[0]
                print('Running {}'.format(mn))
                ent_embs, rel_embs = fetch_embeddings(mt.format(d))
                df_out = parallel_scoring(trip_df, ent_embs, rel_embs)
                df_out.to_csv('ptss-benchmarks/triple_scores_{}_{}_{}.csv'.format(d, mn, ns))
                print(df_out.head())

