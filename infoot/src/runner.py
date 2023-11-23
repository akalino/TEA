import argparse
import datetime
import dateutil
import ot.gromov
import pandas as pd
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from aggregate_data import load_spaces, precompute_cost_matrices, \
    normalize_features, normalize_samples
from infoot import InfoOT, FusedInfoOT


def find(matrix, value):
    value_indexs = [(matrix.index(row), row.index(value))
                    for row in matrix if value == row]
    return value_indexs


def strfdelta(tdelta, fmt):
    d = {"days": tdelta.days}
    d["hours"], rem = divmod(tdelta.seconds, 3600)
    d["minutes"], d["seconds"] = divmod(rem, 60)
    d["ms"] = divmod(d["seconds"], 1000)
    return fmt.format(**d)


def run():
    parser = argparse.ArgumentParser(prog="build_openai_space",
                                        description="Builds sentence embeddings from the OpenAI ADA model")
    parser.add_argument('-s', '--set', required=True, type=str,
                        help='The dataset name', dest='data')
    args = parser.parse_args()
    _check_best = False
    _scale = True
    sizes = [1024, 2048]
    # sizes = [128]
    sim_score_best = []
    was_score_best = []
    source_logger = []
    target_logger = []
    batch_logger = []
    guide_runtime_logger = []
    total_runtime_logger = []
    guide_log = []
    mods = ['openai']
    sources = ['complex', 'conve', 'distmult', 'rescal']
    guide_hyp = [100]
    ptss = True
    for bsz in sizes:
        for trg_name in mods:
            for src_name in sources:
                for ng in guide_hyp:
                    print('Experimenting with triple representation {}, '
                          'sentence representation {}, ' 
                          'number of guides = {}, ' 
                          'and batch size {}'.format(trg_name, src_name, ng, bsz))
                    total_start = datetime.datetime.now()
                    n_guides = ng
                    trips, sents = load_spaces(str(args.data), src_name, trg_name, 5, True, ptss)
                    df = pd.read_csv('t2idx_{}.csv'.format(args.data))
                    l = df[df['rel_idx'] != 0].index.tolist()
                    trips = trips[l]
                    #print(trips.shape)
                    best_dists = {}
                    for_reindex = {}
                    guide_time = []
                    for j in tqdm(range(n_guides)):
                        guide_start = datetime.datetime.now()
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
                            gwd = 1
                        # print(gwd)
                        # should just keep their indices, not the vectors
                        # just do an if else
                        if j == 0:
                            best_dists[gwd] = [trips_norm, sents_norm]
                            for_reindex[gwd] = [tripss, sentss]
                        if gwd < min(list(best_dists.keys())) and j > 0:
                            best_dists = {}
                            for_reindex = {}
                            best_dists[gwd] = [trips_norm, sents_norm]
                            for_reindex[gwd] = [tripss, sentss]
                        guide_end = datetime.datetime.now()
                        guide_search_elapsed = guide_end - guide_start
                        guide_time.append(guide_search_elapsed.total_seconds() * 1000)

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
                        #print(score)
                        #sns.heatmap(score)
                        #plt.show()
                        sims = []
                        new_srcs = []
                        new_trgs = []
                        for j in range(len(projs)):
                            src_smp = projs[j]
                            new_srcs.append(src_smp)
                            trg_smp = trip_proj_trg[j]
                            new_trgs.append(trg_smp)
                            orig = sents_to_proj[j]
                            cos = np.dot(src_smp, trg_smp) / (np.linalg.norm(src_smp)*np.linalg.norm(trg_smp))
                            # print('Cos sim was {} for sample {}'.format(cos, j))
                            sims.append(cos)
                            #if j == bsz - 1:
                            #    df = pd.DataFrame({'source': src_smp,
                            #                       'proj': trg_smp,
                            #                       'target': orig})
                            #    print(df.head())
                        #csp = cosine_similarity(new_srcs, new_trgs)
                        #sns.heatmap(csp)
                        #plt.show()
                        print('Overall similarity was {}'.format(np.mean(sims)))
                        sim_score_best.append(np.mean(sims))
                        print('Best distance was {}'.format(bdists))
                        was_score_best.append(bdists)
                        print('Ran models {} and {}'.format(src_name, trg_name))
                        source_logger.append(src_name)
                        target_logger.append(trg_name)
                        batch_logger.append(bsz)
                        total_end = datetime.datetime.now()
                        guide_log.append(ng)
                        del tripss
                        del sentss
                        del best_dists
                        del for_reindex

                        #total_runtime_logger.append(strfdelta((total_end - total_start), "{hours}:{minutes}:{seconds}"))
                        #guide_runtime_logger.append(strfdelta(np.mean(guide_time), "{hours}:{minutes}:{seconds}"))

                        total_runtime_logger.append((total_end - total_start).total_seconds() * 1000)
                        guide_runtime_logger.append(np.mean(guide_time))

                        pl = False
                        if pl:
                            print('Transportation plan:')
                            print(P)
                            print(P.shape)
                            sns.heatmap(P)
                            plt.show()
                        print(was_score_best[-1])
                        print(total_runtime_logger[-1])
                        print(guide_runtime_logger[-1])
                    except UserWarning:
                        print('Sinkhorn issues, skipping')

    out_df = pd.DataFrame({'source_model': source_logger,
                           'target_model': target_logger,
                           'batch_size': batch_logger,
                           'num_guide': guide_log,
                           'h10': sim_score_best,
                           'was_score': was_score_best,
                           'overall_run': total_runtime_logger,
                           'average_batch_rt': guide_runtime_logger
                           })
    print(out_df)
    if ptss:
        out_df.to_csv('{}_ot_oai_results_wtime.csv'.format(args.data), index=False)
    else:
        out_df.to_csv('{}_ot_oai_results_concat_wtime.csv'.format(args.data), index=False)


if __name__ == "__main__":
    run()
