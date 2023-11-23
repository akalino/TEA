import argparse
import numpy as np
import os
import time
import torch
import matplotlib.pyplot as plt
import ot
import pandas as pd

from tqdm import tqdm
from cuml.decomposition import PCA

from torch.utils.data import DataLoader
from utils import normalize_embeddings, incremental_barycentric_mapping


def read_kg_labels(_dir):
    rid = pd.read_csv(os.path.join(_dir, 'relation_ids.del'), sep="\t", header=None)
    rid_lookup = rid.to_dict(orient='dict')[1]
    rev_rid = {val: key for (key, val) in rid_lookup.items()}
    train_triples = pd.read_csv(os.path.join(_dir, 'train.txt'), sep="\t", header=None)
    train_triples.columns = ['s', 'p', 'o']
    train_triples['pred_idx'] = train_triples['p'].apply(lambda x: rev_rid[x])
    pidx = train_triples['pred_idx'].tolist()
    print('Found {} predicate labels'.format(len(pidx)))
    print(pidx[0:10])
    return pidx


def read_sent_labels(_dir):
    if 'nyt' in _dir:
        sid = pd.read_csv(os.path.join(_dir, 'training_data.csv'))
        sid1 = sid[sid['rel_idx'] != 35]
        sid = pd.read_csv(os.path.join(_dir, 'validation_data.csv'))
        sid2 = sid[sid['rel_idx'] != 35]
        sid = pd.read_csv(os.path.join(_dir, 'testing_data.csv'))
        sid3 = sid[sid['rel_idx'] != 35]
        sidx1 = sid1['rel_idx'].tolist()
        sidx2 = sid2['rel_idx'].tolist()
        sidx3 = sid3['rel_idx'].tolist()
        sidx = sidx1 + sidx2 + sidx3
        print('Found {} sentence labels'.format(len(sidx)))
    else:
        sid = pd.read_csv(os.path.join(_dir, 'train_sentences.txt'), sep='\t')
        sidx1 = sid['predicate_label'].tolist()
        sid = pd.read_csv(os.path.join(_dir, 'hold_sentences.txt'), sep='\t')
        sidx2 = sid['predicate_label'].tolist()
        sid = pd.read_csv(os.path.join(_dir, 'valid_sentences.txt'), sep='\t')
        sidx3 = sid['predicate_label'].tolist()
        sidx = sidx1 + sidx2 + sidx3
        print('Found {} sentence labels'.format(len(sidx)))
        sidx = reindex(sidx)
        print(sidx[0:10])
    return sidx


def reindex(_list):
    """
    Converts string labels to numerical representation.

    :param _list:
    :return:
    """
    uniq_labs = list(set(_list))
    lookup = {}
    for j in range(len(uniq_labs)):
        lookup[uniq_labs[j]] = j
    out_labs = [lookup[x] for x in _list]
    return out_labs


def load_spaces(_ds_name, _kg_name, _lm_name, _ns, _reverse):
    kge_name = 'triples_{}-{}-ht-{}_300_{}.pt'.format(_ds_name,
                                                      _kg_name,
                                                      _ns, _ns)
    ste_name = '{}_{}_space.pt'.format(_ds_name,
                                       _lm_name)
    wd = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    dd = os.path.join(wd, 'data')
    kgp = os.path.join(dd, 'kg-embeddings')
    kgp = os.path.join(kgp, _ds_name)
    kglp = os.path.join(kgp, 'triple_labels')
    _kg_labels = read_kg_labels(kglp)
    lmp = os.path.join(dd, 'sentence-embeddings')
    lmp = os.path.join(lmp, _ds_name)
    lmpp = os.path.join(lmp, 'sentence_labels')
    _sen_labels = read_sent_labels(lmpp)
    _kg_space = torch.load(os.path.join(kgp, kge_name))
    _se_space = torch.load(os.path.join(lmp, ste_name))
    if _reverse:
        return _se_space.cuda(), _kg_space.cuda(), _sen_labels, _kg_labels
    else:
        return _kg_space.cuda(), _se_space.cuda(), _kg_labels, _sen_labels


def train(_args, _reverse):
    torch.manual_seed(_args.seed)
    alpha = _args.alpha
    reg_m = 0.06
    num_workers = _args.worker
    train_bs = _args.batch_size
    max_iterations = _args.max_iterations
    kg_space, se_space, kg_labels, se_labels = load_spaces(_args.dset, _args.s, _args.t, _args.samp, _reverse)
    n_src = se_space.shape[0]
    n_trg = kg_space.shape[0]
    dim_src = se_space.shape[1]
    dim_trg = kg_space.shape[1]

    if dim_src > dim_trg:
        # project src down to trg dim using PCA
        proj = PCA(n_components=dim_trg)
        se_space = torch.tensor(proj.fit_transform(se_space.detach().cpu().numpy())).cuda()
        dim_src = se_space.shape[1]
    elif dim_trg > dim_src:
        proj = PCA(n_components=dim_src)
        kg_space = torch.tensor(proj.fit_transform(kg_space.detach().cpu().numpy())).cuda()
        dim_trg = kg_space.shape[1]

    print(dim_src)
    print(dim_trg)

    se_space, kg_space = normalize_embeddings(_args.p, se_space, kg_space)
    src_idx = [i for i in range(n_src)]
    trg_idx = [j for j in range(n_trg)]

    dset_loaders = {}
    dset_loaders["source"] = DataLoader(src_idx, batch_size=train_bs, shuffle=True, num_workers=num_workers,
                                        drop_last=True)
    dset_loaders["target"] = DataLoader(trg_idx, batch_size=train_bs, shuffle=True,
                                        num_workers=num_workers, drop_last=True)

    all_losses = []
    for i in tqdm(range(max_iterations + 1)):

        # train one iter
        if i % len(dset_loaders["source"]) == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len(dset_loaders["target"]) == 0:
            iter_target = iter(dset_loaders["target"])

        xs = iter_source.next()
        xt = iter_target.next()
        xs, xt = xs.cuda(), xt.cuda()
        # print('===== Batch input shapes =====')
        # print(xs.shape)
        # print(xt.shape)
        # time.sleep(1)

        # print('===== Batched feature shapes =====')
        xs_feat = se_space[xs]
        xt_feat = kg_space[xt]
        # print(xs_feat.shape)
        # print(xt_feat.shape)
        # time.sleep(1)

        # print('===== Ground cost shapes =====')
        M_embed = torch.cdist(xs_feat.double(), xt_feat.double()) ** 2
        # print(M_embed.shape)
        M = alpha * M_embed
        # print('===== Final cost shape =====')
        # print(M.shape)

        # OT computation
        a, b = ot.unif(xs_feat.size()[0]), ot.unif(xt_feat.size()[0])
        try:
            pi = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, M.detach().cpu().numpy(),
                                                         _args.alpha, reg_m=reg_m)
            pi = torch.from_numpy(pi).float().cuda()
            # print('===== OT Shape =====')
            # print(pi.shape)
            # time.sleep(1)
            # print('===== Batched transfer loss =====')
            transfer_loss = 10. * torch.sum(pi * M)
            # print(transfer_loss)
            all_losses.append(transfer_loss.detach().cpu())
        except (RuntimeWarning, UserWarning) as p:
            # print(p)
            pass

    print('===== Final averages loss =====')
    print(np.mean(all_losses))
    print(np.std(all_losses))
    return np.mean(all_losses), np.std(all_losses)


def train_incremental_barycenter(_args, _reverse):
    torch.manual_seed(_args.seed)
    alpha = _args.alpha
    reg_m = 0.06
    num_workers = _args.worker
    train_bs = _args.batch_size
    max_iterations = _args.max_iterations
    kg_space, se_space, kg_labels, se_labels = load_spaces(_args.dset, _args.s, _args.t, _args.samp, _reverse)
    n_src = kg_space.shape[0]
    n_trg = se_space.shape[0]
    dim_src = kg_space.shape[1]
    dim_trg = se_space.shape[1]
    if dim_src > dim_trg:
        # project src down to trg dim using PCA
        proj = PCA(n_components=dim_trg)
        kg_space = torch.tensor(proj.fit_transform(kg_space.detach().cpu().numpy())).cuda()
        dim_src = kg_space.shape[1]
        print('Projected source down to {}'.format(dim_src))
    elif dim_trg > dim_src:
        proj = PCA(n_components=dim_src)
        se_space = torch.tensor(proj.fit_transform(se_space.detach().cpu().numpy())).cuda()
        dim_trg = se_space.shape[1]
        print('Projected target down to {}'.format(dim_trg))

    new_src, new_trg = incremental_barycentric_mapping(kg_space, se_space,
                                                       ot.unif(n_src), ot.unif(n_trg),
                                                       train_bs, train_bs, max_iterations)
    return np.matrix(new_src), np.matrix(new_trg), kg_labels, se_labels


def train_gw(_args, _reverse):
    torch.manual_seed(_args.seed)
    alpha = _args.alpha
    reg_m = 0.06
    num_workers = _args.worker
    train_bs = _args.batch_size
    max_iterations = _args.max_iterations
    kg_space, se_space, kg_labels, se_labels = load_spaces(_args.dset, _args.s, _args.t, _args.samp, _reverse)
    n_src = se_space.shape[0]
    n_trg = kg_space.shape[0]
    dim_src = se_space.shape[1]
    dim_trg = kg_space.shape[1]

    if dim_src > dim_trg:
        # project src down to trg dim using PCA
        proj = PCA(n_components=dim_trg)
        se_space = torch.tensor(proj.fit_transform(se_space.detach().cpu().numpy())).cuda()
        dim_src = se_space.shape[1]
    elif dim_trg > dim_src:
        proj = PCA(n_components=dim_src)
        kg_space = torch.tensor(proj.fit_transform(kg_space.detach().cpu().numpy())).cuda()
        dim_trg = kg_space.shape[1]

    print(dim_src)
    print(dim_trg)

    src_idx = [i for i in range(n_src)]
    trg_idx = [j for j in range(n_trg)]

    dset_loaders = {}
    dset_loaders["source"] = DataLoader(src_idx, batch_size=train_bs, shuffle=True, num_workers=num_workers,
                                        drop_last=True)
    dset_loaders["target"] = DataLoader(trg_idx, batch_size=train_bs, shuffle=True,
                                        num_workers=num_workers, drop_last=True)

    all_losses = []
    for i in tqdm(range(max_iterations + 1)):

        # train one iter
        if i % len(dset_loaders["source"]) == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len(dset_loaders["target"]) == 0:
            iter_target = iter(dset_loaders["target"])

        xs = iter_source.next()
        xt = iter_target.next()
        xs, xt = xs.cuda(), xt.cuda()
        # print('===== Batch input shapes =====')
        # print(xs.shape)
        # print(xt.shape)
        # time.sleep(1)

        # print('===== Batched feature shapes =====')
        xs_feat = se_space[xs]
        xt_feat = kg_space[xt]
        # print(xs_feat.shape)
        # print(xt_feat.shape)
        # time.sleep(1)

        # print('===== Ground cost shapes =====')
        M_embed_src = torch.cdist(xs_feat.double(), xs_feat.double()) ** 2
        M_embed_trg = torch.cdist(xt_feat.double(), xt_feat.double()) ** 2
        # print(M_embed.shape)
        M_src = alpha * M_embed_src
        M_trg = alpha * M_embed_trg
        # print('===== Final cost shape =====')
        # print(M_src)
        # print(M_trg)
        # print(M.shape)
        # time.sleep(10)

        # OT computation
        p, q = ot.unif(xs_feat.size()[0]), ot.unif(xt_feat.size()[0])
        try:
            pi = ot.gromov.entropic_gromov_wasserstein2(M_src.detach().cpu().numpy(),
                                                        M_trg.detach().cpu().numpy(),
                                                        p, q, epsilon=_args.alpha,
                                                        loss_fun="square_loss",)
            # pi = torch.from_numpy(pi).float().cuda()
            # print('===== OT Shape =====')
            # print(pi)
            # time.sleep(10)
            # print('===== Batched transfer loss =====')
            # print(transfer_loss)
            all_losses.append(pi)
        except (RuntimeWarning, UserWarning) as p:
            # print(p)
            pass

    print('===== Final averages loss =====')
    print(np.mean(all_losses))
    print(np.std(all_losses))
    return np.mean(all_losses), np.std(all_losses)


def train_bayes(x):
    torch.manual_seed(x[0])
    alpha = x[1]
    reg_m = x[2]
    num_workers = 8
    train_bs = int(x[3])
    max_iterations = int(x[4])
    kg_space, se_space, kg_labels, se_labels = load_spaces('wikidata', 'rotate', 'gem', 20, False)
    n_src = se_space.shape[0]
    n_trg = kg_space.shape[0]

    dim_src = se_space.shape[1]
    dim_trg = kg_space.shape[1]

    if dim_src != dim_trg:
        # project src down to trg dim using PCA
        proj = PCA(n_components=dim_trg)
        se_space = torch.tensor(proj.fit_transform(se_space.detach().cpu().numpy())).cuda()
        dim_src = se_space.shape[1]

    print(dim_src)
    print(dim_trg)


    src_idx = [i for i in range(n_src)]
    trg_idx = [j for j in range(n_trg)]

    dset_loaders = {}
    dset_loaders["source"] = DataLoader(src_idx, batch_size=train_bs, shuffle=True, num_workers=num_workers,
                                        drop_last=True)
    dset_loaders["target"] = DataLoader(trg_idx, batch_size=train_bs, shuffle=True,
                                        num_workers=num_workers, drop_last=True)

    all_losses = []
    for i in tqdm(range(max_iterations + 1)):

        # train one iter
        if i % len(dset_loaders["source"]) == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len(dset_loaders["target"]) == 0:
            iter_target = iter(dset_loaders["target"])

        xs = iter_source.next()
        xt = iter_target.next()
        xs, xt = xs.cuda(), xt.cuda()
        # print('===== Batch input shapes =====')
        # print(xs.shape)
        # print(xt.shape)
        # time.sleep(1)

        # print('===== Batched feature shapes =====')
        xs_feat = se_space[xs]
        xt_feat = kg_space[xt]
        # print(xs_feat.shape)
        # print(xt_feat.shape)
        # time.sleep(1)

        # print('===== Ground cost shapes =====')
        M_embed = torch.cdist(xs_feat.double(), xt_feat.double()) ** 2
        # print(M_embed.shape)
        M = alpha * M_embed
        # print('===== Final cost shape =====')
        # print(M.shape)

        # OT computation
        a, b = ot.unif(xs_feat.size()[0]), ot.unif(xt_feat.size()[0])
        try:
            pi = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, M.detach().cpu().numpy(),
                                                         alpha, reg_m=reg_m)
            pi = torch.from_numpy(pi).float().cuda()
            # print('===== OT Shape =====')
            # print(pi.shape)
            # time.sleep(1)
            # print('===== Batched transfer loss =====')
            transfer_loss = 10. * torch.sum(pi * M)
            # print(transfer_loss)
            all_losses.append(transfer_loss.detach().cpu())
        except (RuntimeWarning, UserWarning) as p:
            # print(p)
            pass

    print('===== Final averages loss =====')
    print(np.mean(all_losses))
    print(np.std(all_losses))
    return (np.mean(all_losses), np.std(all_losses))


def train_eval_fn(parameterization):
    x = np.array([parameterization.get(f"x{i+1}") for i in range(5)])
    return {"unbalanced_minibatch": train_bayes(x)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unbalanced OT for KG-SE Mapping')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=str, default='rotate', help="Source embedding model")
    parser.add_argument('--t', type=str, default='gem', help="Target embedding model")
    parser.add_argument('--seed', type=int, default=17, help="random seed")
    parser.add_argument('--max_iterations', type=int, default=5000, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=65, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='wikidata',
                        choices=["wikidata", "nyt"])
    parser.add_argument('--samp', type=int, default=30, help='The sample size for PTSS')
    parser.add_argument('--alpha', type=float, default=0.001)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    train(args)
