import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    ds = 'wnrr'
    mods = ['rescal']
    for mn in mods:
        df = pd.read_csv('ptss-benchmarks/{}/triple_scores_{}_{}_emb_5.csv'.format(ds, ds, mn))
        fig, ax = plt.subplots()
        df.hist(column=['sim_score'], figsize=(10, 15),
                backend='matplotlib', ax=ax)
        plt.title('')
        plt.savefig('ptss_{}_{}.png'.format(mn, ds))
        for fr in [0.9]:
            df2 = pd.read_csv('ptss-benchmarks/{}/triple_scores_{}_{}_{}_emb_line.csv'.format(ds, ds, mn, fr))
            df3 = df2.copy()
            fig, ax = plt.subplots()
            df2.hist(column=['pos_score'], figsize=(10, 15), alpha=0.5,
                     backend='matplotlib', ax=ax)
            df3.hist(column=['neg_score'], figsize=(10, 15), alpha=0.5,
                     backend='matplotlib', ax=ax)
            plt.title('')
            # plt.title('Line graph {} '.format(fr) + mn + ' score distribution')
            plt.savefig('slgs_{}_{}.png'.format(ds, fr))
