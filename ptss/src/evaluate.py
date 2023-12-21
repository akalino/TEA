import argparse
import os
#import cudf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

from tqdm import tqdm

from modules.BaselineEvals import find_best_clustering
from modules.BaselineEvals import measure_clusters
from modules.BaselineEvals import hopkins_runner
from modules.BaselineEvals import apply_spatial_historgram
from modules.BaselineEvals import edge_classification, edge_deep


def np2cudf(df):
    # convert numpy array to cuDF dataframe
    df = pd.DataFrame({'fea%d' % i: df[:, i] for i in range(df.shape[1])})
    pdf = cudf.DataFrame()
    for c, column in enumerate(df):
        pdf[str(c)] = df[column]
    return pdf


def get_relational_labels(_ds, _dup):
    if _dup:
        df = pd.read_csv('intermediate/{}_duplicates.csv'.format(_ds))
        _rel_labs = df['p'].tolist()
        _trip_idx = df['triple_idx'].tolist()
    else:
        df = pd.read_csv('intermediate/{}_all_triples_idx.csv'.format(_ds))
        _rel_labs = df['r'].tolist()
        print('Found {} triples'.format(len(_rel_labs)))
        _trip_idx = [j for j in range(len(df))]
        _rel_labs = [_rel_labs[r] for r in range(len(df) - 1)]
    return _rel_labs, _trip_idx


def load_space(_ds_name, _dim, _nwalk, _full_name, _type, _dup):
    wd = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    if 't2v' in _full_name:
        print('Loading triple2vec')
        vecs = np.load('triple_vectors/t2v_{}_{}_{}.npy'.format(_ds_name, _dim, _nwalk),
                       allow_pickle=True)
    else:
        print('Loading pairwise edge similarity vectors')
        pth = os.path.join(wd, 'src', 'triple_vectors_new', '{}.pt'.format(_full_name))
        #pth = os.path.join(wd, 'src', 'train_stats',
        #                   '{}_model.bin'.format(_full_name))
        mod = torch.load(pth)
        vecs = mod.detach().cpu() #['triple_embeddings.weight'].detach().cpu()
        #pth = os.path.join(wd, 'triple_vectors', 'triples_{f}.pt'.format(f=_full_name))
        #vecs = torch.load(pth).detach().cpu()
    #trep_cuda = np2cudf(vecs)
    tgt, idx = get_relational_labels(_ds_name, _dup)
    if _dup:
        try:
            vecs = [vecs[j].numpy() for j in idx]
        except:
            vecs = [vecs[j] for j in idx]
        print('Restricting to {} vectors'.format(len(vecs)))
        vecs = np.asarray(vecs)
    if _type == 'ch':
        out = measure_clusters(vecs.numpy(), tgt)
        return out
    if _type == 'cluster':
        odf = find_best_clustering(vecs, tgt, _full_name)
        hop_mu, hop_sigma = hopkins_runner(vecs.numpy())
        odf['hopkins_mu'] = hop_mu
        odf['hopkins_sigma'] = hop_sigma
        vecs = vecs
        kls_embs, mu_kls, std_kls = apply_spatial_historgram(vecs, 100)
        odf['spat_mu'] = mu_kls
        odf['spat_sigma'] = std_kls
        print(odf)
        return odf
    elif _type == 'edge':
        cl = edge_classification(vecs, tgt)
        print('.8 nodes micro F1 {}'.format(cl[0.8]['micro-f1']))
        dl = edge_deep(vecs, tgt, _dup)
        print('.8 nodes micro F1 {}'.format(dl[0.8]['micro-f1']))
        return cl, dl
    elif _type == 'tsne':
        n_components = 2
        tsne = TSNE(n_components, init='pca', random_state=67)
        tsne_out = tsne.fit_transform(vecs)
        tsne_result_df = pd.DataFrame({'tsne_1': tsne_out[:, 0],
                                       'tsne_2': tsne_out[:, 1],
                                       'label': tgt})
        fig, ax = plt.subplots(1)
        sns.scatterplot(x='tsne_1',
                        y='tsne_2',
                        hue='label', data=tsne_result_df, ax=ax, s=120)
        lim = (tsne_out.min() - 5, tsne_out.max() + 5)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plt.title('TSNE for {}'.format(_full_name))
        plt.savefig('tsne_plots/{}.png'.format(_full_name))


if __name__ == "__main__":
    ds = 'wnrr'
    ns = 0.1
    type = 'edge'
    res = []
    for mn in ['complex',
               'conve',
               'distmult',
               'rescal',
               'rotate',
               'transe']:
        for pt in ['emb']:  #, 'kl', 'freq']:
            full_name = 'triples_{}-{}-ht-{}-{}_{}_{}'.format(ds, mn, ns, pt, ns, pt)
            print('========= Results for {} ========='.format(full_name))
            if type == 'edge':
                cl, cd = load_space(ds, -1, -1, full_name, 'edge', False)
                res.append([full_name,
                            cl[0.8]['micro-f1'],
                            cl[0.8]['macro-f1'],
                            cl[0.8]['weighted-f1'] ,
                            cd[0.8]['micro-f1'],
                            cd[0.8]['macro-f1'],
                            cd[0.8]['weighted-f1'],
                            ])
                df = pd.DataFrame(res)
                df.columns = ['model',
                              'cl-mic',
                              'cl-mac',
                              'cl-wei',
                              'cd-mic',
                              'cd-mac',
                              'cd-wei'
                              ]
                df.to_csv('results/{}_{}_all_nd.csv'.format(ds, ns), index=False)
            # if type == 'cluster':
            #     res = load_space(ds, -1, -1, full_name, 'cluster', False)

