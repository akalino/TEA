import os
import pandas as pd

from tqdm import tqdm


def read_maps(_ds_name):
    wd = os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + os.pardir + os.sep + os.pardir)
    ent_map_path = os.path.join(wd, 'kge/data', _ds_name, 'entity_ids.del')
    ent_map = pd.read_csv(ent_map_path, sep='\t', header=None)
    ent_map.columns = ['index', 'identifier']
    ent_map = ent_map.to_dict()
    ent_map = ent_map['identifier']

    rel_map_path = os.path.join(wd, 'kge/data', _ds_name, 'relation_ids.del')
    rel_map = pd.read_csv(rel_map_path, sep='\t', header=None)
    rel_map.columns = ['index', 'identifier']
    rel_map = rel_map.to_dict()
    rel_map = rel_map['identifier']
    return ent_map, rel_map


if __name__ == '__main__':
    ds = 'fb15k-237'
    mods = ['rotate']
    em, rm = read_maps(ds)
    print(rm)
    wd = os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + os.pardir + os.sep + os.pardir)
    dp = "/kge/data/{}/"
    _lg_ids = pd.read_csv(wd + dp.format(ds) + 'triple_index_df.csv')
    for mn in mods:
        for fr in [0.1, 0.2, 0.3, 0.4]:
            df2 = pd.read_csv('ptss-benchmarks/{}/triple_scores_{}_{}_{}_emb_line.csv'.format(ds, ds, mn, fr))
            df2 = df2[df2['pos_score'] < .25]
            df2 = df2[df2['neg_score'] > .25]
            print('Values in new mass: {}'.format(len(df2)))
            ols = []
            for _, row in tqdm(df2.iterrows()):
                t0 = _lg_ids[_lg_ids['triple'] == row['in']]
                t1 = _lg_ids[_lg_ids['triple'] == row['corr']]
                t2 = _lg_ids[_lg_ids['triple'] == row['out']]
                ols.append(pd.concat([t0, t1, t2]))
            df_out = pd.concat(ols)
            df_out['head_name'] = df_out['head'].apply(lambda x: em[int(x)])
            df_out['tail_name'] = df_out['tail'].apply(lambda x: em[int(x)])
            df_out['rel_name'] = df_out['rel'].apply(lambda x: rm[int(x)])
            df_out.sort_values(['head_name', 'rel_name', 'tail_name'], inplace=True)
            df_out.to_csv('score_overlaps_{}_{}.csv'.format(ds, fr))



