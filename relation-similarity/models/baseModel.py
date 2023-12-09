import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import numpy as np
import os
import time
import math


def fetch_embeddings(_path):
    mod = torch.load(_path, map_location=torch.device('cuda:0'))
    try:
        _ents = mod['model'][0]['_entity_embedder.embeddings.weight'].cpu()
        _rels = mod['model'][0]['_relation_embedder.embeddings.weight'].cpu()
    except KeyError:
        _ents = mod['model'][0]['_entity_embedder._embeddings.weight'].cpu()
        _rels = mod['model'][0]['_relation_embedder._embeddings.weight'].cpu()
    print('Entities {}'.format(_ents.shape))
    print('Relations {}'.format(_rels.shape))
    return _ents, _rels


class baseModel(nn.Module):
    def __init__(self, args, tot_ent, tot_rel):
        super(baseModel, self).__init__()
        self.args = args
        self.ent_emb_size = args['ent_emb_size']
        self.rel_emb_size = args['rel_emb_size']
        self.tot_rel = tot_ent
        self.tot_ent = tot_rel
        self.lr = args['lr']
        self.weight_decay = args['weight_decay']
        self.ent_pretrain = args['ent_pretrain']
        self.rel_pretrain = args['rel_pretrain']
        self.hidden1 = args['hidden1']
        self.hidden2 = args['hidden2']

        self.remb = nn.Embedding(self.tot_rel, self.ent_emb_size)
        self.eemb = nn.Embedding(self.tot_ent, self.rel_emb_size)

        entity_pretrain = []
        relation_pretrain = []
        if self.ent_pretrain:
            try:
                with open(os.path.join(args['input'],
                                           "entity2vec.vec"), 'r') as f:
                    for line in f:
                        line = line.strip().split('\t')
                        entity_pretrain.append([float(ent) for ent in line])
            except FileNotFoundError:
                mod = args['input'].split('/')[2]
                mp = os.path.join(args['input'], '{}-transe.pt'.format(mod))
                entity_pretrain, relation_pretrain = fetch_embeddings(mp)
                print('Ent pretrained: {}'.format(len(entity_pretrain[0])))
            assert(len(entity_pretrain[0]) == args['ent_emb_size'])
            self.eemb.weight = nn.Parameter(torch.FloatTensor(entity_pretrain))
        else:
            nn.init.xavier_uniform_(self.eemb.weight.data)
        if self.rel_pretrain:
            try:
                with open(os.path.join(args['input'],
                                       "relation2vec.vec"), 'r') as f:
                    for line in f:
                        line = line.strip().split('\t')
                        relation_pretrain.append([float(rel) for rel in line])
            except FileNotFoundError:
                mod = args['input'].split('/')[2]
                mc = args['class']
                mp = os.path.join(args['input'], '{}-{}.pt'.format(mod, mc))
                entity_pretrain, relation_pretrain = fetch_embeddings(mp)
                print('Rel pretrained: {}'.format(len(relation_pretrain[0])))
            assert(len(relation_pretrain[0]) == args['rel_emb_size'])
            self.remb.weight = nn.Parameter(torch.FloatTensor(relation_pretrain))
        else:
            nn.init.xavier_uniform_(self.remb.weight.data)

        # self.rProb = nn.Parameter(torch.randn(self.tot_rel))

        self.layers1 = nn.Sequential(nn.Linear(self.rel_emb_size, self.hidden1),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden1, self.hidden2),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden2, self.ent_emb_size))
        
        self.layers2 = nn.Sequential(nn.Linear(self.rel_emb_size, self.hidden1),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden1, self.hidden2),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden2, self.ent_emb_size))

        self.map_layer = nn.Linear(self.ent_emb_size, self.rel_emb_size)

        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)

        self.opt = optim.AdamW(self.parameters(), weight_decay=self.weight_decay, lr=self.lr)
        
    def forward(self, x, layer):
        x = layer(x)
        return torch.matmul(x, self.eemb.weight.transpose(1, 0))

    def train_step(self, heads, rels, tails, train=True):
        print(heads.shape, rels.shape, tails.shape)
        if train:
            print(len(heads))
            tmp1 = self.forward(self.remb(rels), self.layers1)
            print(tmp1.shape)
            print(heads)
            ph = self.log_softmax(tmp1)[[i for i in range(len(heads))], heads]
            if self.ent_emb_size == self.rel_emb_size:
                print('sizes matches')
                tmp2 = self.forward(self.remb(rels) + self.eemb(heads), self.layers2)
                print(tmp2)
            else:
                print('why size mismatch')
                ents_mapped = self.map_layer(self.eemb(heads))
                tmp2 = self.forward(self.remb(rels) + ents_mapped, self.layers2)
                print(tmp2)
            print('did it once')
            print(len(heads))
            print(len(tails))
            print(tmp.shape)
            pt = self.log_softmax(tmp2)[[i for i in range(len(heads))], tails]
            loss = -(torch.sum(ph + pt))
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        else:
            with torch.no_grad():
                tmp1 = self.forward(self.remb(rels), self.layers1)
                ph = self.log_softmax(tmp1)[[i for i in range(len(heads))], heads].cuda()
                if self.ent_emb_size == self.rel_emb_size:
                    tmp2 = self.forward(self.remb(rels) + self.eemb(heads), self.layers2)
                else:
                    ents_mapped = self.map_layer(self.eemb(heads))
                    tmp2 = self.forward(self.remb(rels) + ents_mapped, self.layers2)
                pt = self.log_softmax(tmp2)[[i for i in range(len(heads))], tails].cuda()
                loss = -(torch.sum(ph + pt))

        return loss.item()

    def eval_step(self, rel, sample_ent):
        with torch.no_grad():
            scores = torch.zeros(self.tot_rel).cuda()
            rel = torch.LongTensor([rel for _ in range(sample_ent)]).cuda()
            tmp1 = self.softmax(self.forward(self.remb(rel), self.layers1)).squeeze()
            heads = torch.multinomial(tmp1, 1).squeeze()
            if self.ent_emb_size == self.rel_emb_size:
                tmp2 = self.softmax(self.forward(self.remb(rel) + self.eemb(heads), self.layers2)).squeeze()
            else:
                ents_mapped = self.map_layer(self.eemb(heads))
                tmp2 = self.softmax(self.forward(self.remb(rel) + ents_mapped, self.layers2)).squeeze()
            tails = torch.multinomial(tmp2, 1).squeeze()

            tmp1 = self.forward(self.remb(torch.LongTensor([_ for _ in range(self.tot_rel)]).cuda()), self.layers1)
            for i in range(sample_ent):
                head = torch.LongTensor([heads[i]]*self.tot_rel).cuda()
                tail = torch.LongTensor([tails[i]]*self.tot_rel).cuda()
                tmp1 = self.log_softmax(tmp1 - torch.max(tmp1, dim=-1, keepdim=True)[0])
                ph = tmp1[[_ for _ in range(self.tot_rel)], head].cuda()
                if self.ent_emb_size == self.rel_emb_size:
                    tmp2 = self.forward(self.remb(torch.LongTensor([_ for _ in range(self.tot_rel)]).cuda()) + self.eemb(head), self.layers2)
                else:
                    mp_emb = self.map_layer(self.eemb(head))
                    tmp2 = self.forward(self.remb(torch.LongTensor([_ for _ in range(self.tot_rel)]).cuda()) + mp_emb, self.layers2)
                pt = self.log_softmax(tmp2 - torch.max(tmp2, dim=-1, keepdim=True)[0])[[_ for _ in range(self.tot_rel)], tail].cuda()
                p = ph[rel[0]] + pt[rel[0]] - ph - pt
                scores += p
        return scores            

    ''' 
    For link prediction. This method can be also used to do link prediction.
    We didn't present this part in paper because it seems that the performance
    is not as good as the state-of-the-art.
    '''
    # def rel_predict(self, heads, tails):
    #     with torch.no_grad():
    #         all_rel = torch.LongTensor([_ for _ in range(self.tot_rel)]).cuda()
    #         pr = self.log_softmax(self.rProb).cuda()
    #         tmp1 = self.forward(self.remb(all_rel), self.layers1)
    #         ph = self.log_softmax(tmp1)[:, heads].transpose(1, 0)
    #         batch_size = len(heads)
    #         tile_head = self.eemb(heads).unsqueeze(1).repeat(1, self.tot_rel, 1).view(-1, self.embedding_size)
    #         tmp2 = self.forward(self.remb(all_rel).repeat(batch_size, 1) + tile_head, self.layers2)
    #         tails = tails.unsqueeze(1).repeat(1, self.tot_rel).view(-1, )
    #         first_idx = torch.LongTensor([_ for _ in range(batch_size)]).unsqueeze(1).repeat(1, self.tot_rel).view(-1, )
    #         second_idx = torch.LongTensor([_ for _ in range(self.tot_rel)]).repeat(batch_size)
    #         pt = self.log_softmax(tmp2).view(batch_size, self.tot_rel, -1)[first_idx, second_idx, tails].view(batch_size, -1)
    #         score = pr + ph + pt
    #         return torch.argsort(score)