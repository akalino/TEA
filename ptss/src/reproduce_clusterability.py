import pandas as pd
import numpy as np

from evaluate import load_space, measure_clusters, \
    get_relational_labels


def experiments(_t2v):
    if _t2v:
        ds = ['wnrr', 'fb15k-237']
        dims = [32, 64, 128, 256, 512, 1024]
        print('Loading triple2vec')
        for _ds in ds:
            for _d in dims:
                vecs = np.load('triple_vectors/t2v_{}_{}_{}.npy'.format(_ds,
                                                                        _d,
                                                                        10),
                               allow_pickle=True)
                tgt, idx = get_relational_labels(_ds, False)
                out = measure_clusters(vecs, tgt)
                print('Metrics for {} of dimension {}'.format(_ds, _d))
                print(out)
    else:
        mods = ['complex', 'conve', 'distmult',
                'rescal', 'rotate', 'transe']
        combs = ['ht', 'had', 'avg', 'l1', 'l2']
        ds = ['wnrr', 'fb15k-237']
        all = []
        for m in mods:
            for d in ds:
                for c in combs:
                    full_name = '{}-{}-{}'.format(d, m, c)
                    ch = load_space(d, 0, 0, full_name, 'ch', False)
                    all.append([d, m, c, ch])
        final = pd.DataFrame(all)
        print(final)
        final.to_csv('outputs/clusterbility_metrics_ch.csv', index=False)


if __name__ == "__main__":
    experiments(False)
    experiments(True)

