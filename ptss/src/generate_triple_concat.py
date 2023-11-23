import argparse
import os
import pandas as pd
import torch

from tqdm import tqdm

from modules.TripleConcatenator import TripleConcatenator


def load_triple_indices(_ds_name):
    """
    Loads and maps knowledge graph triples to their respective entity and predicate indices.

    :param _ds_name: Name of the data set to be run.
    :return: Dataframe, all triples.
    """
    wd = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    dp = os.path.join(wd, 'data', _ds_name, 'train.txt')
    train_triples = pd.read_csv(dp, sep='\t', header=None)
    train_triples.columns = ['head', 'relation', 'tail']
    print('Train triples totalled {}'.format(len(train_triples)))

    dp = os.path.join(wd, 'data', _ds_name, 'valid.txt')
    valid_triples = pd.read_csv(dp, sep='\t', header=None)
    valid_triples.columns = ['head', 'relation', 'tail']
    print("Test triples totalled {}".format(len(valid_triples)))

    dp = os.path.join(wd, 'data', _ds_name, 'test.txt')
    test_triples = pd.read_csv(dp, sep='\t', header=None)
    test_triples.columns = ['head', 'relation', 'tail']
    print("Test triples totalled {}".format(len(test_triples)))

    ent_map_path = os.path.join(wd, 'data', _ds_name, 'entity_ids.del')
    ent_map = pd.read_csv(ent_map_path, sep='\t', header=None)
    ent_map.columns = ['index', 'identifier']
    ent_map = ent_map.to_dict()
    ent_map = {v: k for k, v in ent_map['identifier'].items()}

    rel_map_path = os.path.join(wd, 'data', _ds_name, 'relation_ids.del')
    rel_map = pd.read_csv(rel_map_path, sep='\t', header=None)
    rel_map.columns = ['index', 'identifier']
    rel_map = rel_map.to_dict()
    rel_map = {v: k for k, v in rel_map['identifier'].items()}

    train_triples['head_idx'] = train_triples['head'].apply(lambda x: ent_map[x])
    train_triples['tail_idx'] = train_triples['tail'].apply(lambda x: ent_map[x])
    train_triples['rel_idx'] = train_triples['relation'].apply(lambda x: rel_map[x])
    return train_triples


def run(_args):
    mp = os.path.join(os.getcwd(), 'pytorch-models/{}-{}.pt'.format(_args.set, _args.model))
    tc = TripleConcatenator(mp, _args.type)
    tdf = load_triple_indices(_args.set)
    _embs = []
    for r, v in tqdm(tdf.iterrows()):
        h_idx = torch.tensor(v['head_idx'])
        t_idx = torch.tensor(v['tail_idx'])
        r_idx = torch.tensor(v['rel_idx'])
        _embs.append(tc.represent_triple(h_idx, r_idx, t_idx))
    cat_space = torch.stack(_embs)
    print(cat_space.shape)
    torch.save(cat_space, 'concat_triples/{}_{}_concat_triples.pt'.format(_args.set,
                                                                          _args.model))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Contatenated triples')
    parser.add_argument('--model', required=True, help='Seed embedding model')
    parser.add_argument('--type', required=True, help='Type of concatenation')
    parser.add_argument('--set', required=True, help='The benchmark dataset')
    args = parser.parse_args()
    run(args)
