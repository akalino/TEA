import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

FANCY_NAMES = {'complex': 'ComplEx',
               'conve': 'ConvE',
               'distmult': 'DistMult',
               'rescal': 'RESCAL',
               'rotate': 'RotatE',
               'transe': 'TransE'}

if __name__ == "__main__":
    for d in ['fb15k-237']:
        for mn in tqdm(['complex', 'conve', 'distmult',
                   'rescal', 'rotate', 'transe']):
            fig, axs = plt.subplots(1, 4)
            #fig.subplots_adjust(hspace=0.4, wspace=0.4)
            sps = []
            for ns in [5, 10, 20, 30]:
                dp = 'ptss-benchmarks/triple_scores_{}_{}_{}.csv'.format(d, mn, ns)
                df = pd.read_csv(dp)
                if ns == 5:
                    cur_col = 'grey'
                    ax = fig.add_subplot(1, 4, 1)
                elif ns == 10:
                    cur_col = 'blue'
                    ax = fig.add_subplot(1, 4, 2)
                elif ns == 20:
                    cur_col = 'green'
                    ax = fig.add_subplot(1, 4, 3)
                else:
                    cur_col = 'orange'
                    ax = fig.add_subplot(1, 4, 4)

                #ax.axes.get_xaxis().set_visible(False)
                #ax.axes.get_yaxis().set_visible(False)

                df.hist(column=['sim_score'], figsize=(10, 15),
                        backend='matplotlib', ax=ax, color=cur_col)
                ax.set_title('{}, N={}'.format(FANCY_NAMES[mn], ns))
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
            #fig.tight_layout()
            #plt.title('Similarity Score Distributions for {}'.format(mn))
            plt.show()



