from collections import defaultdict
import pandas as pd

from tqdm import tqdm

def index_all_triples(_train_file, _valid_file, _test_file, _entity2id, _relation2id):
    relation = defaultdict(lambda : 0)
    entity = defaultdict(lambda : 0)
    head_list = []
    tail_list = []
    rel_list = []
    triple = []

    with open(_entity2id, 'r') as f:
        for line in f:
            ent, idx = line.split('\t')
            entity[ent] = int(idx)
        entity_num = len(list(entity.keys()))
        print('Found {} entities'.format(entity_num))
    with open(_relation2id, 'r') as f:
        for line in f:
            rel, idx = line.split('\t')
            relation[rel] = int(idx)
        relation_num = len(list(relation.keys()))
        print('Found {} predicates'.format(relation_num))

    nt = 0
    with open(_train_file, 'r') as f:
        for line in tqdm(f):
            nt += 1
            head, rel, tail = line.split("\t")
            tail = tail.strip('\n')
            head_list.append(entity[head])
            tail_list.append(entity[tail])
            rel_list.append(relation[rel])
            triple.append([entity[head], relation[rel], entity[tail]])
    with open(_valid_file, 'r') as f:
        for line in tqdm(f):
            nt += 1
            head, rel, tail = line.split("\t")
            tail = tail.strip('\n')
            head_list.append(entity[head])
            tail_list.append(entity[tail])
            rel_list.append(relation[rel])
            triple.append([entity[head], relation[rel], entity[tail]])
    with open(_test_file, 'r') as f:
        for line in tqdm(f):
            nt += 1
            head, rel, tail = line.split("\t")
            tail = tail.strip('\n')
            head_list.append(entity[head])
            tail_list.append(entity[tail])
            rel_list.append(relation[rel])
            triple.append([entity[head], relation[rel], entity[tail]])
    triple = pd.DataFrame(triple)
    triple.columns = ['h', 'r', 't']
    return triple

if __name__ == "__main__":
    trips = index_all_triples('train.txt', 'valid.txt', 'test.txt', 'entity2id.txt', 'relation2id.txt')
    trips.to_csv('all_triples_idx.csv', index=False)
