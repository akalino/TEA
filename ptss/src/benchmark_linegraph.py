import pickle
import pandas as pd
import torch
import os
import numpy as np
import random

from tqdm import tqdm

def load_triple_map(_kg):
    wd = os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + os.pardir + os.sep + os.pardir)
    dp = "/kge/data/{}/"
    basepath = wd + dp.format(_kg) + 'triple-arrays.pkl'
    with open(basepath, 'rb') as f:
        _t2id = pickle.load(f)
    _lg_edges = pd.read_csv(wd + dp.format(_kg) + 'line_graph.csv', header=None)
    _lg_edges.columns = ['in', 'out']
    return _t2id, _lg_edges


def fetch_embeddings(_path):
    mod = torch.load(_path, map_location=torch.device('cuda:0'))
    try:
        _ents = mod['model'][0]['_entity_embedder.embeddings.weight'].cpu()
        _rels = mod['model'][0]['_relation_embedder.embeddings.weight'].cpu()
    except KeyError:
        _ents = mod['model'][0]['_entity_embedder._embeddings.weight'].cpu()
        _rels = mod['model'][0]['_relation_embedder._embeddings.weight'].cpu()
    return _ents, _rels


def score_triples(_t1, _t2, _ent_embs, _rel_embs, _triples_idx):
    e1 = _triples_idx[_t1]
    e2 = _triples_idx[_t2]
    h1 = _ent_embs[e1[0]].unsqueeze(0)
    h2 = _ent_embs[e2[0]].unsqueeze(0)
    r1 = _rel_embs[e1[1]].unsqueeze(0)
    r2 = _rel_embs[e2[1]].unsqueeze(0)
    t1 = _ent_embs[e1[2]].unsqueeze(0)
    t2 = _ent_embs[e2[2]].unsqueeze(0)
    h_sim = torch.cosine_similarity(h1, h2, ).detach().numpy()
    r_sim = torch.cosine_similarity(r1, r2).detach().numpy()
    t_sim = torch.cosine_similarity(t1, t2).detach().numpy()
    _score = np.mean([h_sim, r_sim, t_sim])
    return _score


def parallel_score(_triple_df, _ent_embs, _rel_embs):
    num_cores = multiprocessing.cpu_count() - 1
    num_partitions = num_cores
    df_split = np.array_split(_triple_df, num_partitions)
    pool = multiprocessing.Pool(num_cores)
    df = pd.concat(pool.map(apply_scoring, df_split))
    pool.close()
    pool.join()
    return df

if __name__ == "__main__":
    kg = 'wnrr'
    model_paths = ['pytorch-models/{}-complex.pt',
                   'pytorch-models/{}-conve.pt',
                   'pytorch-models/{}-distmult.pt',
                   'pytorch-models/{}-rescal.pt',
                   'pytorch-models/{}-rotate.pt',
                   'pytorch-models/{}-transe.pt'
                   ]
    t2id, lg_edges = load_triple_map('wnrr')
    all_edges = np.amax(lg_edges)
    print('Max edges: {}'.format(all_edges))
    edge_ids = [x for x in range(all_edges)]
    for m in model_paths:
        ents, rels = fetch_embeddings(m.format(kg))
        scores = []
        truths = []
        for _, row in tqdm(lg_edges.iterrows()):
            scores.append(score_triples(row['in'], row['out'], ents, rels, t2id))
            truths.append(1)
            sub = lg_edges[lg_edges['in'] == row['in']]
            true_list = sub['out'].tolist()
            corr_list = [x for x in edge_ids if x not in true_list]
            scores.append(score_triples(row['in'], random.choice(corr_list), ents, rels, t2id))
            truths.append(-1)
        lg_edges['score'] = scores
        lg_edges['truths'] = truths
        mod = m.split('-')[-1].split('.')[0]
        lg_edges.to_csv('ptss-benchmarks/{g}/triple_scores_{g}_{m}_emb_line.csv'.format(kg, mod))

