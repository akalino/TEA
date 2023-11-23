import subprocess

import pandas as pd
import numpy as np

from evaluate import load_space, measure_clusters, \
    get_relational_labels


def experiments(_t2v, _dup):
    if _t2v:
        ds = ['wikidata']
        dims = [32, 64, 128, 256, 512, 1024]
        print('Loading triple2vec')
        all = []
        for _ds in ds:
            for _d in dims:
                el, eh = load_space(_ds, _d, 10, 't2v', 'edge', _dup)
                all.append(['t2v', _ds, _d, el, eh])
        out = pd.DataFrame(all)
        print(out)
        if _dup:
            v = 'restricted'
        else:
            v = 'all'
        out.to_csv('outputs/edge_classification_t2v_{}.csv'.format(v), index=False)

    else:
        mods = [#'complex',
                #'conve',
                #'distmult',
                #'rescal',
                'rotate',
                #'transe'
                ]
        combs = ['ht']  #, 'had', 'avg', 'l1', 'l2']
        ds = ['fb15k-237']
        ns = [5, 10, 20, 30]
        emb_dim = 300
        all = []
        for m in mods:
            for d in ds:
                for c in combs:
                    for ss in ns:
                        full_name = 'triples_{}-{}-{}-{}_{}_{}'.format(d, m, c, ss, emb_dim, ss)
                        print(full_name)
                        try:
                            el, eh = load_space(d, 0, 0, full_name, 'edge', _dup)
                            all.append([d, m, c, el, eh, ss])
                        except FileNotFoundError:
                            try:
                                call_str = "gsutil -m cp gs://triple_vectors/{}.pt".format(full_name)
                                target_loc = "triple_vectors/{}.pt".format(full_name)
                                subprocess.call("{} {}".format(call_str, target_loc), shell=True)
                                el, eh = load_space(d, 0, 0, full_name, 'edge', _dup)
                                all.append([d, m, c, el, eh])
                            except FileNotFoundError:
                                print('Passing on model {}'.format(full_name))
        final = pd.DataFrame(all)
        print(final)
        if _dup:
            v = 'restricted'
        else:
            v = 'all'
        final.to_csv('outputs/edge_classification_{}_{}.csv'.format(d, v), index=False)


if __name__ == "__main__":
    #experiments(False, True)
    experiments(False, False)
    #experiments(True, True)
    #experiments(True, False)

