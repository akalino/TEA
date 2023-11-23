import torch
import os
import json
import pandas as pd

from tqdm import tqdm


class SupervisedDataLoader(object):

    def __init__(self, _train_path, _valid_path, _batch_sz):
        self.train_path = _train_path
        self.valid_path = _valid_path
        pl = _train_path.split("/")[0:-1]
        pl.append("t2idx_nytfb.csv")
        self.triple_path = "/".join(pl)

        pl2 = _train_path.split("/")[0:-1]
        pl2.append('entity_mapper.json')
        self.entity_path = "/".join(pl2)

        pl3 = _train_path.split("/")[0:-1]
        pl3.append('relation_mapper.json')
        self.rel_path = "/".join(pl3)

        self.batch_size = _batch_sz
        self.train_x, self.train_y, self.valid_x, self.valid_y = self.process()

    def process(self):
        triple_idx = pd.read_csv(self.triple_path)
        triple_idx.columns = ['triple_id', 'head', 'relation', 'tail',
                              'head_idx', 'tail_idx', 'rel_idx']
        with open(self.entity_path, 'r') as f:
            em = json.load(f)
        with open(self.rel_path, 'r') as g:
            rm = json.load(g)
        heads = []
        tails = []
        rels = []
        trips = []
        for row, vals in tqdm(triple_idx.iterrows()):
            v = vals['head'].replace(".", "/").strip()
            v = "/" + v
            head_idx = em[v]
            b = vals['tail'].replace(".", "/").strip()
            b = "/" + b
            tail_idx = em[b]
            rel = vals['relation'].strip()
            rel_idx = rm[rel]
            heads.append(head_idx)
            tails.append(tail_idx)
            rels.append(rel_idx)
            trips.append(vals['triple_id'])
        triple_df = pd.DataFrame({'triple_id': trips,
                                  'em1_idx': heads,
                                  'rel_idx': rels,
                                  'em2_idx': tails})
        triple_df = triple_df[triple_df['rel_idx'] != 35]
        triple_df['left-key'] = triple_df.apply(lambda x: str(x.em1_idx) + '-' + str(x.em2_idx) + '-' + str(x.rel_idx), axis=1)
        triple_df = triple_df[['left-key', 'triple_id']]
        train_df = pd.read_csv(self.train_path)
        train_df = train_df[train_df['rel_idx'] != 35]

        train_df['right-key'] = train_df.apply(
            lambda x: str(x.em1_idx) + '-' + str(x.em2_idx) + '-' + str(x.rel_idx), axis=1)
        train_df = train_df[['right-key', 'sent_id']]
        new_df = triple_df.merge(train_df, how='left', left_on='left-key', right_on='right-key')
        #new_df = triple_df.merge(train_df, on=['rel_idx', 'em1_idx', 'em2_idx'])
        #new_df = triple_df.merge(train_df, how='left',
        #                         left_on=['em1_idx_l', 'rel_idx_l', 'em2_idx_l'],
        #                         right_on=['em1_idx', 'rel_idx', 'em2_idx'])
        new_df = new_df[~new_df['sent_id'].isnull()]
        #new_df = new_df[['triple_id', 'rel_idx',
        #                 'em1_idx', 'em2_idx',
        #                 'sent_id']]

        new_df['sent_id'] = new_df['sent_id'].astype(int)
        new_df['triple_id'] = new_df['triple_id'].astype(int)
        new_df = new_df.sample(frac=.1, random_state=17)

        train_df_x = new_df[['triple_id']]
        train_df_y = new_df[['sent_id']]
        train_x = torch.from_numpy(train_df_x.values)
        train_y = torch.from_numpy(train_df_y.values)

        valid_df = pd.read_csv(self.valid_path)
        valid_df = valid_df[valid_df['rel_idx'] != 35]

        valid_df['right-key'] = valid_df.apply(
            lambda x: str(x.em1_idx) + '-' + str(x.em2_idx) + '-' + str(x.rel_idx), axis=1)

        newer_df = triple_df.merge(valid_df, how='left', left_on='left-key', right_on='right-key')
        #newer_df = triple_df.merge(valid_df, on=['rel_idx', 'em1_idx', 'em2_idx'])
        #newer_df = triple_df.merge(valid_df, how='inner',
        #                           left_on=['em1_idx_l', 'rel_idx_l', 'em2_idx_l'],
         #                          right_on=['em1_idx', 'rel_idx', 'em2_idx'])
        newer_df = newer_df[~newer_df['sent_id'].isnull()]
        newer_df['sent_id'] = newer_df['sent_id'].astype(int)
        newer_df['triple_id'] = newer_df['triple_id'].astype(int)
        newer_df = newer_df.sample(frac=.1, random_state=17)

        valid_df_x = newer_df[['triple_id']]
        valid_df_y = newer_df[['sent_id']]
        valid_x = torch.from_numpy(valid_df_x.values)
        valid_y = torch.from_numpy(valid_df_y.values)
        print('Running with {} training and {} validation samples'.format(len(new_df),
                                                                          len(newer_df)))
        return train_x, train_y, valid_x, valid_y

    def get_validation(self):
        return self.valid_x, self.valid_y

    def get_validation_batches(self):
        valid_batch_x = []
        valid_batch_y = []
        permutation = torch.randperm(self.valid_x.size()[0])
        for i in range(0, self.valid_x.size()[0], self.batch_size):
            indices = permutation[i:i + self.batch_size]
            valid_batch_x.append(self.valid_x[indices])
            valid_batch_y.append(self.valid_y[indices])
        return valid_batch_x, valid_batch_y

    def create_all_batches(self):
        permutation = torch.randperm(self.train_x.size()[0])
        batch_x = []
        batch_y = []
        for i in range(0, self.train_x.size()[0], self.batch_size):
            indices = permutation[i:i + self.batch_size]
            batch_x.append(self.train_x[indices])
            batch_y.append(self.train_y[indices])
        return batch_x, batch_y
