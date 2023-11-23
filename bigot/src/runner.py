import ot.gromov
import pandas as pd
import numpy as np
import torch

from tqdm import tqdm

from aggregate_data import load_spaces, precompute_cost_matrices, \
    normalize_features, normalize_samples
from infoot import InfoOT, FusedInfoOT


def find(matrix, value):
    value_indexs = [(matrix.index(row), row.index(value))
                    for row in matrix if value == row]
    return value_indexs


def run():
    _check_best = True
    _scale = True
    sizes = [512, 1024, 2048]
    sim_score_best = []
    was_score_best = []
    source_logger = []
    target_logger = []
    batch_logger = []
    mods = ['complex', 'conve', 'rescal', 'rotate', 'transe', 'distmult']
    for bsz in sizes:
        for src_name in mods:
            print('Experimenting with {} triple '
                  'representation and batch size {}'.format(src_name, bsz))
            n_guides = 100
            trg_name = 'dct_1'
            trips, sents = load_spaces('nytfb', src_name, trg_name, 5, True)
            df = pd.read_csv('t2idx_nytfb.csv')
            l = df[df['rel_idx'] != 0].index.tolist()
            trips = trips[l]
            #print(trips.shape)
            best_dists = {}
            for_reindex = {}
            for j in range(n_guides):
                trips_samp = trips[torch.randperm(trips.shape[0])]
                sents_samp = sents[torch.randperm(sents.shape[0])]
                tripss = trips_samp[0:bsz].cpu().detach().numpy()
                sentss = sents_samp[0:bsz].cpu().detach().numpy()

                trips_norm, sents_norm, trip_scale, sent_scale = normalize_features(tripss, sentss)

                if _scale:
                    trip_cost, sent_cost = precompute_cost_matrices(sents_norm, trips_norm, 'knn', False)
                    sents_to_proj = normalize_samples(sents[bsz+1:bsz*2+1].cpu().detach().numpy(),
                                                      sent_scale)
                    trip_proj_trg = normalize_samples(trips[bsz+1:bsz*2+1].cpu().detach().numpy(),
                                                      trip_scale)
                else:
                    sents_to_proj = sents[bsz+1:bsz*2+1]
                    trip_proj_trg = trips[bsz+1:bsz*2+1]
                    trip_cost, sent_cost = precompute_cost_matrices(sentss, tripss, 'knn', False)

                try:
                    gwd = ot.gromov.gromov_wasserstein2(sent_cost, trip_cost,
                                                        ot.unif(sent_cost.shape[0]),
                                                        ot.unif(trip_cost.shape[0]),
                                                        log=False, verbose=False)
                except UserWarning:
                    print('skipping due to non-convergence of GW distance')
                    pass
                # print(gwd)
                best_dists[gwd] = [trips_norm, sents_norm]
                for_reindex[gwd] = [tripss, sentss]

            bdists = min(list(best_dists.keys()))
            tripss, sentss = best_dists[bdists]
            trip_ind, sent_ind = for_reindex[bdists]

            if _check_best:
                trips = trips.cpu().detach().numpy()
                sents = sents.cpu().detach().numpy()
                trip_idx = []
                for r in trip_ind:
                    trip_idx.append(int(np.argwhere((r == trips).all(1))[0]))
                sent_idx = []
                for r in sent_ind:
                    sent_idx.append(int(np.argwhere((r == sents).all(1))[0]))
                tbv = pd.DataFrame({'source_indices': trip_idx,
                                    'target_indices': sent_idx})
                tbv.to_csv('sanity_logs/{}_{}_{}.csv'.format(src_name, trg_name, bsz))

            try:
                mapper = FusedInfoOT(Xs=sentss, Xt=tripss,
                                     h=0.35, Ys=None, lam=0.2, reg=0.01, init='gw')
                # 1/15 changed reg to 0.01 from 0.1
                P = mapper.solve(numIter=100)

                projs = mapper.project(sents_to_proj,
                                       method='conditional')
                score = mapper.conditional_score(sents_to_proj)
                sims = []
                for j in range(len(projs)):
                    src_smp = projs[j]
                    trg_smp = trip_proj_trg[j]
                    orig = sents_to_proj[j]
                    cos = np.dot(src_smp, trg_smp) / (np.linalg.norm(src_smp)*np.linalg.norm(trg_smp))
                    # print('Cos sim was {} for sample {}'.format(cos, j))
                    sims.append(cos)
                    #if j == bsz - 1:
                    #    df = pd.DataFrame({'source': src_smp,
                    #                       'proj': trg_smp,
                    #                       'target': orig})
                    #    print(df.head())
                print('Overall similarity was {}'.format(np.mean(sims)))
                sim_score_best.append(np.mean(sims))
                print('Best distance was {}'.format(bdists))
                was_score_best.append(bdists)
                print('Ran models {} and {}'.format(src_name, trg_name))
                source_logger.append(src_name)
                target_logger.append(trg_name)
                batch_logger.append(bsz)
            except UserWarning:
                print('Sinkhorn issues, skipping')
    out_df = pd.DataFrame({'source_model': source_logger,
                           'target_model': target_logger,
                           'batch_size': batch_logger,
                           'sim_score': sim_score_best,
                           'was_score': was_score_best
                           })
    print(out_df)
    out_df.to_csv('{}_ot_results.csv'.format(trg_name), index=False)


if __name__ == "__main__":
    run()
