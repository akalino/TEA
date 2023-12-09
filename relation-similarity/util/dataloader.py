import copy
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

from tqdm import tqdm


class relationDataset(Dataset):
    def __init__(self, train_file, entity2id, relation2id):
        self.relation = defaultdict(lambda : 0)
        self.entity = defaultdict(lambda : 0)
        self.head = []
        self.tail = []
        self.rel = []
        self.triple = defaultdict(list)
        with open(entity2id, 'r') as f:
            # self.entity_num = int(f.readline().strip())
            for line in f:
                ent, idx = line.split('\t')
                self.entity[ent] = int(idx)
            self.entity_num = len(list(self.entity.keys()))
            print('Found {} entities'.format(self.entity_num))
        with open(relation2id, 'r') as f:
            # self.relation_num = int(f.readline().strip())
            for line in f:
                rel, idx = line.split('\t')
                self.relation[rel] = int(idx)
            self.relation_num = len(list(self.relation.keys()))
            print('Found {} predicates'.format(self.relation_num))

        nt = 0
        with open(train_file, 'r') as f:
            # self.triple_num = int(f.readline().strip())
            for line in tqdm(f):
                nt += 1
                head, rel, tail = line.split("\t")
                tail = tail.strip('\n')
                self.head.append(self.entity[head])
                self.tail.append(self.entity[tail])
                self.rel.append(self.relation[rel])
                # self.rel.append(int(rel))
                self.triple[self.relation[rel]].append([self.entity[head], self.entity[tail]])
            self.head = np.array(self.head)
            self.rel = np.array(self.rel)
            self.tail = np.array(self.tail)
            self.triples_num = nt
            
        print("Relation nums:%d"%len(self.relation))
        print("Entity nums:%d"%len(self.entity))
        print("Triple nums:%d"%self.triples_num)

    def __len__(self):
        return self.triples_num

    def __getitem__(self, index):
        return self.head[index], self.rel[index], self.tail[index], 1