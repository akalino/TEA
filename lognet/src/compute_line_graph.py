import numpy as np
import pandas as pd
from typing import Tuple, Set
import itertools
from joblib import Parallel, delayed
import multiprocessing
import os
from time import time
import csv as csv
import pickle

num_cores = multiprocessing.cpu_count()

def add_edges(source_node_ID, all_triples):
    column = 0
    mask = np.in1d(all_triples[:, column], [source_node_ID])
    ##we need to add edges between the ids of the triple in triple_IDs
    triple_IDs = np.argwhere(mask==True)
    edgesAdded: Set[Tuple] = []

    for x in itertools.product(triple_IDs, triple_IDs):
        if x[0] != x[1]:
            edge: Tuple[str, str] = (x[0].item()), (x[1].item())
            edgesAdded.append(edge)
    return edgesAdded

def write_triple_line_graph(results, KG, basepath):
    csv_f = basepath + '/line_graph.csv'
    with open(csv_f,'w') as csv_file:
        writer = csv.writer(csv_file)
        for res in results:
            for r in res:
                writer.writerow(r)


def computeLineGraph(alltriples, numEntities, KG, basepath):
    start = time()
    results = Parallel(n_jobs=num_cores + 2, verbose=10, prefer="threads")(delayed(add_edges)(subjectID,alltriples) for subjectID in range(0,numEntities+1))
    print("Time to compute line graph {:.6f} s".format(time() - start))
    write_triple_line_graph(results,KG,basepath)


def main():
    wd = os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + os.pardir + os.sep + os.pardir)
    dp = "/kge/data/wnrr/"
    basepath = wd + dp

    KG = "wnrr"
    triples = wd + dp + 'all_triples_idx.csv'
    all_triples = pd.read_csv(triples)
    all_triples = all_triples.to_numpy()

    d = dict(enumerate((all_triples[:, [0,1,2]]), 0))
    csv_f = basepath + 'triples2ids.csv'
    with open(csv_f, 'w') as csv_file:
        print(d, file=csv_file)
    with open(basepath + 'triple-arrays.pkl', 'wb') as f:
        pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)

    num_entities = np.amax(all_triples)
    print("Max entity id=",num_entities)
    computeLineGraph(alltriples=all_triples,numEntities=num_entities,KG=KG,basepath=basepath)


if __name__ == "__main__":
    main()