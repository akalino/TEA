import argparse
import json
import networkx as nx
import numpy as np
import os
import pandas as pd

from time import time
from tqdm import tqdm


def load_triple_graph(_fp, _g):
    """

    :param _fp: File path to all triples.
    :return: Graph object from triples.
    """
    #if _g == 'wnrr':
    #    df = pd.read_csv(_fp, skiprows=1)
    #    df.columns = ['h', 't', 'r']
    #else:
    df = pd.read_csv(_fp)
    G = nx.from_pandas_edgelist(df, 's', 'o', edge_attr='p', create_using=nx.DiGraph())
    print('Build graph')
    #print(list(G.edges)[0:10])
    idg_list = list(G.in_degree())
    odg_list = list(G.out_degree())
    print('Average in-degree: {}'.format(np.mean(idg_list)))
    print('Average out-degree: {}'.format(np.mean(odg_list)))

    cc = nx.number_connected_components(G.to_undirected())
    print('Number components: {}'.format(cc))
    scc = nx.number_strongly_connected_components(G)
    print('Number strong components: {}'.format(scc))
    wcc = nx.number_weakly_connected_components(G)
    print('Number weak components: {}'.format(wcc))
    apl = nx.average_shortest_path_length(G.to_undirected())
    print('Average shortest path length: {}'.format(apl))
    idg = {}
    odg = {}
    print('Building degree features')
    for k in tqdm(range(len(idg_list))):
        idg[idg_list[k][0]] = idg_list[k][1]
        odg[odg_list[k][0]] = odg_list[k][1]
    print('Building shortest path map')
    start = time()
    sp = dict(nx.all_pairs_shortest_path(G))
    print("Time to shortest paths {:.6f} s".format(time() - start))
    start2 = time()
    spl = {}
    for st in tqdm(sp.keys()):
        pl = {}
        for ed in sp[st].keys():
            pl[ed] = len(sp[st][ed])
        spl[st] = pl
    print("Time to SLP {:.6f} s".format(time() - start2))
    return sp, spl, idg, odg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sampling strategies for PTSS')
    parser.add_argument('--graph', required=True, help='Name of the target graph')
    args = parser.parse_args()
    wd = os.path.normpath(os.getcwd())
    print(wd)
    wdi = os.path.join(wd, 'intermediate')
    fn = '{}_triples.csv'.format(args.graph)
    fp = os.path.join(wdi, fn)
    #if g == 'wnrr':
    #    dp = "/data/wnrr/"
    #    fp = wd + dp + 'triples.csv'
    #else:
    #    dp = "/data/FB15K237/unpacked/"
    #    fp = wd + dp + "all_triples_idx.csv"
    short, spl_dict, idg, odg = load_triple_graph(fp, None)
    with open(wd + dp + 'sp.json', 'w') as fp:
        json.dump(short, fp)
    with open(wd + dp + 'spl.json', 'w') as fp:
        json.dump(spl_dict, fp)
    with open(wd + dp + 'idg.json', 'w') as fp:
        json.dump(idg, fp)
    with open(wd + dp + 'odg.json', 'w') as fp:
        json.dump(odg, fp)
