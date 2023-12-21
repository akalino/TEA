import pandas as pd
import pickle
import numpy as np
import math
import torch
import torch.nn as nn
import os
import json
import wandb

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sentence_transformers.readers import InputExample
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers import losses
from sentence_transformers.util import fullname

from modules.TripleEmbedderOrig import TripleEmbedder
from modules.CustomCosineLoss import CustomCosineSimilarityLoss


BATCH_SIZE = 128
# Batch size for WN18RR 128
EPOCHS = 50
LR = 2e-3
# LR for WN18RR 2e-2
SENT_DIM = 300
PATIENCE = 1


def create_emb_layer(weights_matrix):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    emb_layer.weight.requires_grad = True
    return emb_layer, num_embeddings, embedding_dim


class DenseEncoder(nn.Module):

    def __init__(self, init_weights, in_features, out_features, bias,
                 output_path, full_name):
        super(DenseEncoder, self).__init__()
        self.full_name = full_name
        self.out_path = output_path
        self.triple_embeddings, self.num_emb, self.emb_dim = create_emb_layer(init_weights)
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.activation = nn.Tanh()
        self.linear = nn.Linear(self.in_features, self.out_features, self.bias)

    def get_config_dict(self):
        return {'in_features': self.in_features,
                'out_features': self.out_features, 'bias': self.bias,
                'activation_function': fullname(self.activation)}

    def forward(self, trip1, trip2):
        a = self.triple_embeddings(trip1)
        b = self.triple_embeddings(trip2)
        _mapped1 = self.activation(self.linear(a))
        _mapped2 = self.activation(self.linear(b))
        return _mapped1, _mapped2

    def eval_forward(self, trip1):
        a = self.triple_embeddings(trip1)
        return a

    def tokenize(self, _repr):
        return _repr

    def save(self, output_path):
        with open(os.path.join(self.out_path,
                               '{}_config.json'.format(self.full_name)), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut)
        torch.save(self.state_dict(), os.path.join(self.out_path,
                                                   '{}_model.bin'.format(self.full_name)))


class DenseExpander(nn.Module):

    def __init__(self, init_weights, in_features, out_features, bias,
                 output_path, full_name):
        super(DenseExpander, self).__init__()
        self.full_name = full_name
        self.out_path = output_path
        self.triple_embeddings, self.num_emb, self.emb_dim = create_emb_layer(init_weights)
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.activation = nn.Tanh()
        self.linear = nn.Linear(self.in_features, self.out_features, self.bias)

    def get_config_dict(self):
        return {'in_features': self.in_features,
                'out_features': self.out_features,
                'bias': self.bias,
                'activation_function': fullname(self.activation)}

    def forward(self, trip1, trip2):
        a = self.triple_embeddings(trip1)
        b = self.triple_embeddings(trip2)
        _mapped1 = self.activation(self.linear(a))
        _mapped2 = self.activation(self.linear(b))
        return _mapped1, _mapped2

    def tokenize(self, _repr):
        return _repr

    def save(self, output_path):
        with open(os.path.join(self.out_path,
                               '{}_config.json'.format(self.full_name)), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut)
        torch.save(self.state_dict(), os.path.join(self.out_path,
                                                   '{}_model.bin'.format(self.full_name)))


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
            nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def fetch_embeddings(_path):
    print(_path)
    mod = torch.load(_path, map_location=torch.device('cuda:0'))
    #set_requires_grad(mod, True)
    try:
        _ents = mod['model'][0]['_entity_embedder.embeddings.weight']
        _rels = mod['model'][0]['_relation_embedder.embeddings.weight']
    except KeyError:
        _ents = mod['model'][0]['_entity_embedder._embeddings.weight']
        _rels = mod['model'][0]['_relation_embedder._embeddings.weight']
    return _ents, _rels


def represent_triples(_h, _t, _rel, _ent_embs, _rel_embs, _concat_method, _norm):
    if _concat_method == 'ht':
        _triple = torch.cat([_ent_embs[_h], _ent_embs[_t]], dim=0)
    elif _concat_method == 'add':
        _triple = _ent_embs[_h] + _ent_embs[_t]
    elif _concat_method == 'avg':
        _triple = (_ent_embs[_h] + _ent_embs[_t]) / 2
    elif _concat_method == 'had':
        _triple = _ent_embs[_h] * _ent_embs[_t]
    elif _concat_method == 'l1':
        _triple = torch.abs(
            _ent_embs[_h] - _ent_embs[_t])
    elif _concat_method == 'l2':
        _triple = torch.abs(
            _ent_embs[_h] - _ent_embs[_t])**2
    if _norm:
        _triple = nn.functional.normalize(_triple, p=2, dim=0)
    return _triple.clone().detach().requires_grad_(True)


def data_loader(_d, _mn, _ns, _concat_method, _pt, _norm, _induct):
    train_samples = []
    dev_samples = []
    test_samples = []
    df = pd.read_csv('ptss-benchmarks/{}/triple_scores_{}_{}_{}_{}.csv'.format(_d, _d, _mn, _pt, _ns))
    ent_embs, rel_embs = fetch_embeddings('pytorch-models/{}-{}.pt'.format(_d, _mn))
    emb_dim = ent_embs.shape[1]
    if _concat_method == 'ht':
        feat_dim = emb_dim * 2
    else:
        feat_dim = emb_dim
    total = len(df)
    if _induct:
        train_size = total
    else:
        print('Total samples: {}'.format(total))
        train_size = int(.8*total)
        print('Total train: {}'.format(train_size))
        dev_size = int(.1*total) + train_size
        print('Total dev: {}'.format(dev_size))
    j = 0
    triple_dict = {}
    for k, row in tqdm(df.iterrows()):
        j += 1
        score = float(row['sim_score'])  # / 5.0  # Normalize score to range 0 ... 1
        trip_1 = represent_triples(row['true_head'], row['true_tail'], row['true_rel'],
                                   ent_embs, rel_embs, _concat_method, _norm)
        trip_2 = represent_triples(row['head_idx'], row['tail_idx'], row['rel_idx'],
                                   ent_embs, rel_embs, _concat_method, _norm)
        if row['index'] not in triple_dict.keys():
            triple_dict[row['index']] = trip_2
        if row['triple_index_1'] not in triple_dict.keys():
            triple_dict[row['triple_index_1']] = trip_1

        inp_example = InputExample(texts=[torch.tensor(row['triple_index_1']), torch.tensor(row['index'])], label=score)
        if _induct:
            train_samples.append(inp_example)
        else:
            if j < train_size:
                train_samples.append(inp_example)
            elif (j > train_size) and (j < dev_size):
                dev_samples.append(inp_example)
            else:
                test_samples.append(inp_example)
    print('Added {} training, {} dev and {} test'.format(len(train_samples), len(dev_samples), len(test_samples)))
    print('Indexed {} triples'.format(len(list(triple_dict.keys()))))
    weight_est = list(triple_dict.values())
    vec_dim = weight_est[0].shape[0]
    print('Training on vectors of {} dimension'.format(vec_dim))
    weights_out = np.zeros((len(weight_est), vec_dim))
    for j in range(len(weight_est)):
        weights_out[j, :] = weight_est[j].cpu().detach().numpy()
    print('Weights shape: {}'.format(weights_out.shape))
    return train_samples, dev_samples, test_samples, feat_dim, weights_out, triple_dict


def data_loader_lg(_d, _mn, _ns, _concat_method, _pt, _norm, _induct):
    train_samples = []
    dev_samples = []
    test_samples = []
    df = pd.read_csv('ptss-benchmarks/{}/triple_scores_{}_{}_{}_{}_line.csv'.format(_d, _d, _mn, _ns, _pt))
    wd = os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + os.pardir + os.sep + os.pardir)
    dp = "/kge/data/{}/"
    _lg_edges = pd.read_csv(wd + dp.format(_d) + 'line_graph.csv', header=None)
    _lg_edges.columns = ['in', 'out']
    num_triples = np.amax(_lg_edges)

    ent_embs, rel_embs = fetch_embeddings('pytorch-models/{}-{}.pt'.format(_d, _mn))
    emb_dim = ent_embs.shape[1]
    if _concat_method == 'ht':
        feat_dim = emb_dim * 2
    else:
        feat_dim = emb_dim
    total = len(df)
    if _induct:
        train_size = total
    else:
        print('Total samples: {}'.format(total))
        train_size = int(.8*total)
        print('Total train: {}'.format(train_size))
        dev_size = int(.1*total) + train_size
        print('Total dev: {}'.format(dev_size))
    j = 0
    triple_dict = {}
    wd = os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + os.pardir + os.sep + os.pardir)
    dp = "/kge/data/{}/"
    basepath = wd + dp.format(_d) + 'triple-arrays.pkl'
    with open(basepath, 'rb') as f:
        _t2id = pickle.load(f)
    for k, row in tqdm(df.iterrows()):
        j += 1
        lgt1 = row['in']
        lgt2 = row['out']
        lgtn = row['corr']
        pos_score = float(row['pos_score'])
        neg_score = float(row['neg_score'])
        t1 = _t2id[lgt1]
        t2 = _t2id[lgt2]
        t3 = _t2id[lgtn]
        trip_1 = represent_triples(t1[0], t1[2], t1[1], ent_embs, rel_embs, _concat_method, _norm)
        trip_2 = represent_triples(t2[0], t2[2], t2[1], ent_embs, rel_embs, _concat_method, _norm)
        trip_3 = represent_triples(t3[0], t3[2], t3[1], ent_embs, rel_embs, _concat_method, _norm)
        if row['in'] not in triple_dict.keys():
            triple_dict[int(row['in'])] = trip_1
        if row['out'] not in triple_dict.keys():
            triple_dict[int(row['out'])] = trip_2
        if row['corr'] not in triple_dict.keys():
            triple_dict[int(row['corr'])] = trip_3

        pos_example = InputExample(texts=[torch.tensor(int(row['in'])), torch.tensor(int(row['out']))], label=pos_score)
        neg_example = InputExample(texts=[torch.tensor(int(row['in'])), torch.tensor(int(row['corr']))], label=neg_score)
        if _induct:
            train_samples.append(pos_example)
            train_samples.append(neg_example)
        else:
            if j < train_size:
                train_samples.append(pos_example)
                train_samples.append(neg_example)
            elif (j > train_size) and (j < dev_size):
                dev_samples.append(pos_example)
                dev_samples.append(neg_example)
            else:
                test_samples.append(pos_example)
                test_samples.append(neg_example)
    print('Added {} training, {} dev and {} test'.format(len(train_samples), len(dev_samples), len(test_samples)))
    print('Indexed {} triples'.format(len(list(triple_dict.keys()))))

    weights_out = np.zeros((num_triples+1, feat_dim))
    print('Training on vectors of {} dimension'.format(weights_out.shape))


    for j in triple_dict.keys():
        weights_out[j, :] = triple_dict[j].cpu().detach().numpy()

    print('Weights shape: {}'.format(weights_out.shape))
    return train_samples, dev_samples, test_samples, feat_dim, weights_out, triple_dict


def training(_train, _dev, _test, _feat_dim, _init_w, _num_epochs, _full_name, _mt, _sd, _ns, _pt):
    # run_tracker = wandb.init()
    train_batch_size = BATCH_SIZE
    sentence_dimension = _sd
    print('Training model type {}-{}-{}-{}'.format(_mt, _sd, _ns, _pt))
    model_save_path = 'train_stats'
    try:
        wd = os.path.normpath(os.getcwd())
        model_out_path = os.path.join(wd, 'triple_vectors_new',
                                      'triples_{f}_{d}_{n}_{p}.pt'.format(f=_full_name, d=sentence_dimension, n=_ns, p=_pt))
        print(model_out_path)
        vecs = torch.load(model_out_path)
        print('Already trained this model, loaded vectors with shape {}'.format(vecs.shape))
    except FileNotFoundError:
        if _mt == 'standard':
            mod = DenseEncoder(_init_w, _feat_dim, _feat_dim*2, True, model_save_path, _full_name)
        elif _mt == 'match_sent':
            w = torch.empty(_init_w.shape[0], sentence_dimension)
            _init_w = torch.nn.init.xavier_normal_(w)
            mod = DenseEncoder(_init_w, sentence_dimension, _feat_dim*2, True, model_save_path, _full_name)
        model = TripleEmbedder(modules=[mod], patience=PATIENCE)
        # run_tracker.watch(model)
        train_dataloader = DataLoader(_train, shuffle=True, batch_size=train_batch_size)
        train_loss = CustomCosineSimilarityLoss(model=model)
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(_dev, name=_full_name)
        print('Testing after {} iterations'.format(int(len(_dev)) / 2))
        warmup_steps = math.ceil(len(train_dataloader) * _num_epochs * 0.1)  # 10% of train data for warm-up
        # Train the model
        model.fit(train_objectives=[(train_dataloader, train_loss)],
                  evaluator=evaluator,
                  epochs=_num_epochs,
                  optimizer_params={'lr': LR},
                  evaluation_steps=int(len(_dev) / 2),
                  warmup_steps=warmup_steps,
                  output_path=model_save_path)
                  # experiment_tracker=run_tracker)
        if _mt == 'standard':
            triple_embs = mod.triple_embeddings.weight
            print(triple_embs.shape)
            wd = os.path.normpath(os.getcwd())
            model_out_path = os.path.join(wd, 'triple_vectors_new',
                                          'triples_{f}_{n}_{p}.pt'.format(f=_full_name, n=_ns, p=_pt))
            torch.save(triple_embs, model_out_path)
        elif _mt == 'match_sent':
            triple_embs = mod.triple_embeddings.weight
            print(triple_embs.shape)
            wd = os.path.normpath(os.getcwd())
            model_out_path = os.path.join(wd, 'triple_vectors_new',
                                          'triples_{f}_{d}_{n}_{p}.pt'.format(f=_full_name, d=sentence_dimension,
                                                                              n=_ns, p=_pt))
            torch.save(triple_embs, model_out_path)
        # run_tracker.finish(quiet=True)
    return model_out_path


def run_triple_fitting(_mod_name, _feature_name, _ds_name, _mt, _sd, _ns, _pt):
    print('Working with {} data'.format(_ds_name))
    full_name = '{}-{}-{}-{}-{}'.format(_ds_name, _mod_name, _feature_name, _ns, _pt)
    try:
        wd = os.path.normpath(os.getcwd())
        model_out_path = os.path.join(wd, 'triple_vectors_new',
                                      'triples_{f}_{n}_{p}.pt'.format(f=full_name, n=_ns, p=_pt))
        vecs = torch.load(model_out_path)
        print('Already trained this model, loaded vectors with shape {}'.format(vecs.shape))

    except FileNotFoundError:
        try:
            weights = np.load('intermediate/{}-weights.npy'.format(full_name), allow_pickle=True)
            print('Loaded weights')
            train = np.load('intermediate/{}-training.npy'.format(full_name), allow_pickle=True)
            print('Loaded training triples')
            dev = np.load('intermediate/{}-dev.npy'.format(full_name), allow_pickle=True)
            print('Loaded dev triples')
            test = np.load('intermediate/{}-test.npy'.format(full_name), allow_pickle=True)
            print('Loaded test triples')
            feature_dim = weights.shape[1]
            print('Training on {} features'.format(feature_dim))
        except FileNotFoundError:
            # train, dev, test, feature_dim, weights, trip_dict = data_loader(_ds_name, _mod_name, _ns,
            #                                                                 _feature_name, _pt, False, False)
            train, dev, test, feature_dim, weights, trip_dict = data_loader_lg(_ds_name, _mod_name, _ns,
                                                                               _feature_name, _pt, False, False)
            np.save('intermediate/{}-training.npy'.format(full_name), train)
            print('wrote train')
            np.save('intermediate/{}-dev.npy'.format(full_name), dev)
            print('wrote dev')
            np.save('intermediate/{}-test.npy'.format(full_name), test)
            print('wrote test')
            np.save('intermediate/{}-weights.npy'.format(full_name), weights)
            print('wrote weights')
        weights = torch.tensor(weights)
        print(weights.shape)
        model_out_path = training(train, dev, test, feature_dim, weights, EPOCHS, full_name, _mt, _sd, _ns, _pt)
    return model_out_path


if __name__ == "__main__":
    run_triple_fitting('conve', 'ht', 'wnrr', 'standard', 300, 20)

