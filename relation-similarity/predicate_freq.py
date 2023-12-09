import numpy as np
import collections
import json
import torch
import os
import pandas as pd
import sys


def read_predicate_map(_path):
    """
    Loads predicate to index map.

    :param _path: Path to text file.
    :return: Dictionary with indices as keys, string names as values.
    """
    _map = {}
    with open(_path) as f:
        for line in f:
            vs = line.split("\t")
            if len(vs) > 1:
                _map[vs[1].strip("\n")] = vs[0]
    return _map


def read_all_triples(_set):
    wd = os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + os.pardir)
    train_path = os.path.join(wd, 'GRAPHS', _set, 'train.txt')
    valid_path = os.path.join(wd, 'GRAPHS', _set, 'valid.txt')
    test_path = os.path.join(wd, 'GRAPHS', _set, 'test.txt')
    train_df = pd.read_csv(train_path, sep="\t")
    train_df.columns = ['head', 'pred', 'tail']
    valid_df = pd.read_csv(valid_path, sep="\t")
    valid_df.columns = ['head', 'pred', 'tail']
    test_df = pd.read_csv(test_path, sep="\t")
    test_df.columns = ['head', 'pred', 'tail']
    train_preds = train_df['pred'].tolist()
    valid_preds = valid_df['pred'].tolist()
    test_preds = test_df['pred'].tolist()
    all_preds = train_preds + valid_preds + test_preds
    frequency = collections.Counter(all_preds)
    print(frequency)
    print(dict(frequency))

    try:
        mp = os.path.join(wd, 'GRAPHS', _set, 'relation2id.txt')
        map = read_predicate_map(mp)
    except FileNotFoundError:
        mp = os.path.join(wd, 'GRAPHS', _set, 'relation_ids.del')
        map = read_predicate_map(mp)


if __name__ == "__main__":
    read_all_triples('wnrr')
