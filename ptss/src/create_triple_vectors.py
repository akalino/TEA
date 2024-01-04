import timeit

from train import run_triple_fitting


def experiments():
    mods = [
            'transe',
            ]
    combs = ['ht']
    ds = ['fb15k-237']
    mt = 'standard'
    pred_type = ['emb']
    sds = [300]  # [768, 1024, 2048, 4096, 4800]
    ns = [30]
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

