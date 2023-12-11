import numpy as np
import os
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from time import time


def calculate_relatedness(_fp):
    """
    Calculates the predicate relatedness of a graph.
    :param _fp: File path to triples.
    :return: Matrix of predicate-to-predicate similarities.
    """
    start = time()
    df = pd.read_csv(_fp)
    df.sort_values('r', inplace=True)
    groups = df.groupby('r')
    # print(groups.groups.keys())
    n_preds = len(groups)
    rel_matrix = np.zeros((n_preds, n_preds))
    tf = [(np.log(1 + pd.merge(groups.get_group(j), groups.get_group(i), how='right', on=['h', 't']).dropna().shape[0]))
          for i in groups.groups.keys() for j in groups.groups.keys()]
    print("Time to compute TF {:.6f} s".format(time() - start))
    start2 = time()
    pos = [(i, j) for i in range(0, n_preds) for j in range(0, n_preds)]
    rows, cols = zip(*pos)
    rel_matrix[rows, cols] = tf
    idf_values = {i: np.log(n_preds / np.count_nonzero(rel_matrix[i])) for i in range(0, n_preds)}
    tf_idf = [rel_matrix[i][j] * idf_values[j] for i in range(0, n_preds) for j in range(0, n_preds)]
    rel_matrix[rows, cols] = tf_idf
    print("Time to compute IDF {:.6f} s".format(time() - start2))
    start3 = time()
    similarity = [cosine_similarity(rel_matrix[i].reshape(1, n_preds), rel_matrix[j].reshape(1, n_preds))[0][0] + 0.001
                  for i in range(0, n_preds) for j in range(0, n_preds)]
    rel_matrix[rows, cols] = similarity
    print("Time to compute similarities {:.6f} s".format(time() - start3))
    return rel_matrix


if __name__ == "__main__":
    wd = os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + os.pardir)
    print(wd)
    dp = "/kge/data/fb15k-237/"
    fp = wd + dp + "all_triples_idx.csv"
    rm = calculate_relatedness(fp)
    np.savetxt(wd + dp + "rel_matrix.csv", rm, fmt='%1.3f', delimiter=",")
    np.save('fb15k-237-freq.npy', rm)
