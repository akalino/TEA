import argparse
import numpy as np
import ot.gromov
import pandas as pd
import torch

from trainer.aggregate_data import load_spaces, precompute_cost_matrices, \
    normalize_features, normalize_samples
from trainer.infoot import InfoOT, FusedInfoOT


def single_bigot_trial(_triples, _sentences, _bsz, _dataset, _kge_name, _sen_name):
    """
    Runs one trial of Batched Information Guided Optimal Transport.

    :param _triples: Embedding space of triples.
    :param _sentences: Embedding space of sentences.
    :param _bsz: Batch size.
    :param _dataset: Dataset name.
    :param _kge_name: Name of KG embedding model.
    :param _sen_name: Name of sentence embedding model.
    :return: List: [major metrics to be logged]
    """
    sim_score_best = []
    was_score_best = []
    source_logger = []
    target_logger = []
    batch_logger = []
    _scale = True
    _check_best = False
    n_guides = 100
    best_dists = {}
    for_reindex = {}
    for j in range(n_guides):
        trips_samp = _triples[torch.randperm(_triples.shape[0])]
        sents_samp = _sentences[torch.randperm(_sentences.shape[0])]
        tripss = trips_samp[0:_bsz].cpu().detach().numpy()
        sentss = sents_samp[0:_bsz].cpu().detach().numpy()
        trips_norm, sents_norm, trip_scale, sent_scale = normalize_features(tripss, sentss)
        if _scale:
            trip_cost, sent_cost = precompute_cost_matrices(sents_norm, trips_norm, 'knn', False)
            sents_to_proj = normalize_samples(_sentences[_bsz + 1:_bsz * 2 + 1].cpu().detach().numpy(),
                                              sent_scale)
            trip_proj_trg = normalize_samples(_triples[_bsz + 1:_bsz * 2 + 1].cpu().detach().numpy(),
                                              trip_scale)
        else:
            sents_to_proj = _sentences[_bsz + 1:_bsz * 2 + 1]
            trip_proj_trg = _triples[_bsz + 1:_bsz * 2 + 1]
            trip_cost, sent_cost = precompute_cost_matrices(sentss, tripss, 'knn', False)

        try:
            gwd = ot.gromov.gromov_wasserstein2(sent_cost, trip_cost,
                                                ot.unif(sent_cost.shape[0]),
                                                ot.unif(trip_cost.shape[0]),
                                                log=False, verbose=False)
        except UserWarning:
            print('skipping due to non-convergence of GW distance')
            pass
        best_dists[gwd] = [trips_norm, sents_norm]
        for_reindex[gwd] = [tripss, sentss]

    bdists = min(list(best_dists.keys()))
    tripss, sentss = best_dists[bdists]
    trip_ind, sent_ind = for_reindex[bdists]

    if _check_best:
        trips = _triples.cpu().detach().numpy()
        sents = _sentences.cpu().detach().numpy()
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
            cos = np.dot(src_smp, trg_smp) / (np.linalg.norm(src_smp) * np.linalg.norm(trg_smp))
            # print('Cos sim was {} for sample {}'.format(cos, j))
            sims.append(cos)
            # if j == bsz - 1:
            #    df = pd.DataFrame({'source': src_smp,
            #                       'proj': trg_smp,
            #                       'target': orig})
            #    print(df.head())
        print('Overall similarity was {}'.format(np.mean(sims)))
        sim_score_best.append(np.mean(sims))
        print('Best distance was {}'.format(bdists))
        was_score_best.append(bdists)
        print('Ran models {} and {}'.format(_kge_name, _sen_name))
        source_logger.append(_kge_name)
        target_logger.append(_sen_name)
        batch_logger.append(_bsz)
    except UserWarning:
        print('Sinkhorn issues, skipping')
    _metrics = [source_logger, target_logger, batch_logger,
                sim_score_best, was_score_best]
    return _metrics


def run(_args):
    """
    A function for running a BIGOT experimental trial.

    :param _args: The input arguments to parse.
    :return: None, outputs training metrics.
    """
    src_path = _args.source_path[0]
    trg_path = _args.target_path[0]
    dataset = _args.dataset[0]
    kge_methods = ['distmult', 'complex', 'conve',
                   'rescal', 'rotate', 'transe']
    seb_methods = ['glove', 'gem', 'sbert', 'random',
                   'skipthought', 'quickthought',
                   'laser', 'infersentv1', 'infersentv2']
    batch_sizes = [256, 512, 1024, 2048]
    sources = []
    targets = []
    batches = []
    gwds = []
    sims = []
    for bsz in batch_sizes:
        for kge in kge_methods:
            for sen in seb_methods:
                triple_path = src_path + '/' + '{}-{}-5.pt'.format(dataset,
                                                                   kge)
                triples = torch.load(triple_path)
                sent_path = trg_path + '/' + '{}-{}.pt'.format(dataset,
                                                               sen)
                sentences = torch.load(sent_path)
                print('Triple space shape: {}'.format(triples.shape))
                print('Sentence space shape: {}'.format(sentences.shape))
                run_metrics = single_bigot_trial(triples, sentences, bsz, dataset, kge, sen)
    logframe = pd.DataFrame({'sources': sources,
                             'targets': targets,
                             'batch': batches,
                             'gwd': gwds,
                             'a_sims': sims})
    op = _args.job_dir + '/' + '{}-metrics.csv'.format(dataset)
    logframe.to_csv(op, index=False)
