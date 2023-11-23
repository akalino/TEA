from train import run_triple_fitting


def experiments():
    mods = [
            'distmult'
            ]
    combs = ['ht']
    ds = ['nytfb']
    mt = 'match_sent'
    sds = [300]  # [768, 1024, 2048, 4096, 4800]
    ns = [5]
    for d in ds:
        for m in mods:
            for c in combs:
                for s in sds:
                    for n in ns:
                        op = run_triple_fitting(m, c, d, mt, s, n)


if __name__ == "__main__":
    experiments()

