import os
import pandas as pd

from tqdm import tqdm


def load_triple_indices(_ds_name):
    """
    Loads and maps knowledge graph triples to their respective entity and predicate indices.

    :param _ds_name: Name of the data set to be run.
    :return: Dataframe, all triples.
    """
    wd = os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + os.pardir + os.sep + os.pardir)
    dp = os.path.join(wd, 'kge/data', _ds_name, 'train.txt')
    train_triples = pd.read_csv(dp, sep='\t', header=None)
    train_triples.columns = ['head', 'relation', 'tail']

    dp = os.path.join(wd, 'kge/data', _ds_name, 'valid.txt')
    valid_triples = pd.read_csv(dp, sep='\t', header=None)
    valid_triples.columns = ['head', 'relation', 'tail']

    dp = os.path.join(wd, 'kge/data', _ds_name, 'test.txt')
    test_triples = pd.read_csv(dp, sep='\t', header=None)
    test_triples.columns = ['head', 'relation', 'tail']

    ent_map_path = os.path.join(wd, 'kge/data', _ds_name, 'entity_ids.del')
    ent_map = pd.read_csv(ent_map_path, sep='\t', header=None)
    ent_map.columns = ['index', 'identifier']
    ent_map = ent_map.to_dict()
    ent_map = {v: k for k, v in ent_map['identifier'].items()}

    rel_map_path = os.path.join(wd, 'kge/data', _ds_name, 'relation_ids.del')
    rel_map = pd.read_csv(rel_map_path, sep='\t', header=None)
    rel_map.columns = ['index', 'identifier']
    rel_map = rel_map.to_dict()
    rel_map = {v: k for k, v in rel_map['identifier'].items()}

    train_triples['head_idx'] = train_triples['head'].apply(lambda x: ent_map[x])
    train_triples['tail_idx'] = train_triples['tail'].apply(lambda x: ent_map[x])
    train_triples['rel_idx'] = train_triples['relation'].apply(lambda x: rel_map[x])
    return train_triples


def generate_samples(_df, _d):
    """
    Applies the PTSS sampling methodology from paper section 3.

    :param _df: Data frame of all triples.
    :param _d: Name of dataset.
    :return: None, saves samples triple pairs to csv in ptss-benchmarks directory.
    """
    all_pairs = []
    for r, v in tqdm(_df.iterrows()):
        candidates = []
        same_head_only = _df[(_df['head_idx'] == v['head_idx']) &
                             (_df['rel_idx'] != v['rel_idx']) &
                             (_df['tail_idx'] != v['tail_idx'])]
        same_head_only.reset_index(inplace=True, drop=False)
        if len(same_head_only) <= 5:
            candidates.append(same_head_only)
        else:
            candidates.append(same_head_only.sample(n=5, random_state=17))
        same_tail_only = _df[(_df['head_idx'] != v['head_idx']) &
                             (_df['rel_idx'] != v['rel_idx']) &
                             (_df['tail_idx'] == v['tail_idx'])]
        same_tail_only.reset_index(inplace=True, drop=False)
        if len(same_tail_only) <= 5:
            candidates.append(same_tail_only)
        else:
            candidates.append(same_tail_only.sample(n=5, random_state=17))
        same_rel_only = _df[(_df['head_idx'] != v['head_idx']) &
                            (_df['rel_idx'] == v['rel_idx']) &
                            (_df['tail_idx'] != v['tail_idx'])]
        same_rel_only.reset_index(inplace=True, drop=False)
        if len(same_rel_only) <= 5:
            candidates.append(same_rel_only)
        else:
            candidates.append(same_rel_only.sample(n=5, random_state=17))
        no_commonalities = _df[(_df['head_idx'] != v['head_idx']) &
                               (_df['rel_idx'] != v['rel_idx']) &
                               (_df['tail_idx'] != v['tail_idx'])]
        no_commonalities.reset_index(inplace=True, drop=False)
        if len(no_commonalities) <= 5:
            candidates.append(no_commonalities)
        else:
            candidates.append(no_commonalities.sample(n=5, random_state=17))
        cand = pd.concat(candidates)
        cand['true_head'] = v['head_idx']
        cand['true_rel'] = v['rel_idx']
        cand['true_tail'] = v['tail_idx']
        cand['triple_index_1'] = r
        all_pairs.append(cand)
    full_samples = pd.concat(all_pairs)
    full_samples.to_csv('ptss-benchmarks/triple_benchmarks_{}.csv'.format(_d))


if __name__ == "__main__":
    ds = ['fb15k-237']
    for d in ds:
        triples = load_triple_indices(d)
        generate_samples(triples, d)
