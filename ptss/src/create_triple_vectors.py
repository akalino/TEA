import timeit

from train import run_triple_fitting


def experiments():
    mods = [
            'complex',
            'conve',
            'distmult',
            'rescal',
            'rotate',
            'transe'
            ]
    combs = ['ht']
    ds = ['wnrr']
    mt = 'standard'
    pred_type = ['freq'] #, 'freq', 'kl']
    sds = [300]  # [768, 1024, 2048, 4096, 4800]
    ns = [5]
    for d in ds:
        for m in mods:
            for c in combs:
                for s in sds:
                    for p in pred_type:
                        for n in ns:
                            start = timeit.default_timer()
                            op = run_triple_fitting(m, c, d, mt, s, n, p)
                            stop = timeit.default_timer()
                            print('Time: ', stop - start)


if __name__ == "__main__":
    experiments()

