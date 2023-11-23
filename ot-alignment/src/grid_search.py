import argparse
import pandas as pd
import os
import torch
import warnings

from unbalanced_minibatch import train_gw, train_incremental_barycenter
from utils import plot_spaces


def gen_args(_src, _trg, _seed, _alpha, _prep):
    parser = argparse.ArgumentParser(description='Unbalanced OT for KG-SE Mapping')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=str, default=_src, help="Source embedding model")
    parser.add_argument('--t', type=str, default=_trg, help="Target embedding model")
    parser.add_argument('--seed', type=int, default=_seed, help="random seed")
    parser.add_argument('--p', type=str, default=_prep, help='Type of vector preprocessing')
    parser.add_argument('--max_iterations', type=int, default=5000, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=1024, help="batch_size")
    parser.add_argument('--worker', type=int, default=8, help="number of workers")
    parser.add_argument('--dset', type=str, default='wikidata',
                        choices=["wikidata", "nyt"])
    parser.add_argument('--samp', type=int, default=20, help='The sample size for PTSS')
    parser.add_argument('--alpha', type=float, default=_alpha)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    return args


def log_runs():
    prior_runs = os.listdir('wikidata_outputs')
    sources = [
               'complex',
               'conve',
               'distmult',
               'rescal',
               'rotate',
               'transe']
    targets = [
               #'dct1', 'dct2', 'dct3',
               'gem',
               'glove',
               #'infersent1glove', 'infersent2ft',
               'laser', 'random', 'sentbert'
               ]
    seeds = [17]
    alphas = [1e-3, 1e-4]
    preps = [None]
    results = []
    for s in sources:
        for t in targets:
            for d in seeds:
                for a in alphas:
                    for p in preps:
                        run_name = '{}-{}-{}-{}.png'.format(s, t, d, a)
                        if run_name in prior_runs:
                            print('Skipping experiment {}'.format(run_name))
                            pass
                        else:
                            print('Training src: {}, trg: {}, seed: {}, alpha: {}, prep: {}'.format(s, t, d, a, p))
                            args = gen_args(s, t, d, a, p)
                            new_src, new_trg, src_labs, trg_labs = train_incremental_barycenter(args, False)
                            print(new_src.shape)
                            print(new_trg.shape)
                            plot_spaces(args.dset, new_src, new_trg, src_labs, trg_labs, d, s, t, a)
                            #print('Solving for transport plan')
                            #tp = torch.linalg.solve(torch.tensor(new_src), torch.tensor(new_trg))
                            #print(tp.shape)
                            #mean_loss, std_loss = train_incremental_barycenter(args, True)
                            #results.append([s, t, d, a, p, mean_loss, std_loss])
        #odf = pd.DataFrame(results)
        #odf.to_csv('results/trials_gw.csv', index=False)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    log_runs()
