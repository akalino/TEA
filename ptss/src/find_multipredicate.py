import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm


def load_triple_indices(_ds_name):
    wd = os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + os.pardir + os.sep + os.pardir)
    dp = "/kge/data/{}/".format(_ds_name)
    basepath = wd + dp
    train = basepath + 'train.txt'
    valid = basepath + 'valid.txt'
    test = basepath + 'test.txt'
    train_triples = pd.read_csv(train, sep='\t', header=None)
    valid_triples = pd.read_csv(valid, sep='\t', header=None)
    test_triples = pd.read_csv(test, sep='\t', header=None)
    # all_trip = pd.concat([train_triples, valid_triples, test_triples],
    #                ignore_index=True)
    all_trip = train_triples
    all_trip.columns = ['head', 'relation', 'tail']
    # all_trip = train_triples
    print('{} has {} total triples'.format(_ds_name, len(all_trip)))
    
    ent_map_path = os.path.join(basepath, 'entity_ids.del')
    ent_map = pd.read_csv(ent_map_path, sep='\t', header=None)
    ent_map.columns = ['index', 'identifier']
    ent_map = ent_map.to_dict()
    ent_map = {v: k for k, v in ent_map['identifier'].items()}

    rel_map_path = os.path.join(basepath, 'relation_ids.del')
    rel_map = pd.read_csv(rel_map_path, sep='\t', header=None)
    rel_map.columns = ['index', 'identifier']
    rel_map = rel_map.to_dict()
    rel_map = {v: k for k, v in rel_map['identifier'].items()}

    all_trip['s'] = all_trip['head'].apply(lambda x: ent_map[x])
    all_trip['o'] = all_trip['tail'].apply(lambda x: ent_map[x])
    all_trip['p'] = all_trip['relation'].apply(lambda x: rel_map[x])
    all_trip = all_trip[['s', 'p', 'o']]
    all_trip.to_csv('intermediate/{}_triples.csv'.format(_ds_name), index=False)
    return all_trip


def identify_triples(_df, _ds_name):
    total = len(_df)
    _df.reset_index(drop=False, inplace=True)
    _df.columns = ['triple_idx', 's', 'p', 'o']
    num_preds = _df.groupby(['s', 'o']).p.agg('count')
    num_preds = num_preds.reset_index(name='count', drop=False)
    dup_edges = num_preds[num_preds['count'] > 1]
    iter_me = dup_edges[['s', 'o']].drop_duplicates()
    preds = []
    trip_idx = []
    for idx, row in tqdm(iter_me.iterrows()):
        res1 = _df[_df['s'] == row['s']]
        res2 = res1[_df['o'] == row['o']]
        pp = res2.p.tolist()
        ids = res2.triple_idx.tolist()
        for i in ids:
            trip_idx.append(i)
        for v in pp:
            preds.append([row['s'], row['o'], v])
    all_dup_triples = pd.DataFrame(preds)
    all_dup_triples.columns = ['s', 'o', 'p']
    print('Found {} triples with duplicate '
         'edges of {} total triples,'
         'ratio of {}'.format(len(all_dup_triples),
                              total,
                              float(len(all_dup_triples) / total)))
    all_dup_triples['triple_idx'] = trip_idx
    all_dup_triples.to_csv('intermediate/{}_duplicates.csv'.format(_ds_name), index=False)
    return dup_edges, all_dup_triples


if __name__ == "__main__":
    ds = ['wnrr', 'fb15k-237']
    for d in ds:
        df = load_triple_indices(d)
        identify_triples(df, d)
