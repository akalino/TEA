from evaluate import load_space


def run():
    ds = ['wnrr', 'fb15k-237']
    models = ['transe', 'rescal', 'conve',
              'distmult', 'rotate', 'complex']
    aggs = ['ht', 'avg', 'had', 'l1', 'l2']
    for d in ds:
        for m in models:
            for f in aggs:
                full_name = '{}-{}-{}'.format(d, m, f)
                load_space(d, 256, 10, full_name, 'tsne', True)


if __name__ == "__main__":
    run()

