# PTSS
This repository contains the data and code to support the experiments found in "Repurposing Knowledge Graph Embeddings for Triple Representation via Weak Supervision".


## Getting Started
To begin, clone and enter this repository, then clone and install LibKGE as detailed here (https://github.com/uma-pi1/kge).

After installation, cd to data and run ```download_kgs.sh```.

Next, cd to ```src/pytorch_models``` and download the pre-trained models for WN18RR and FB15K-237 found here (https://github.com/uma-pi1/kge#results-and-pretrained-models).

## Reproducing Experiments

Change directories to ```src``` and exceute the steps in the following order to reproduce our experiments.

1. Run the sampling strategy for finding candidate triple pairs ```python apply_sampling_strategy.py```.
2. Build wealky supervised PTSS scores by running ```python benchmark_ptss.py```. The results will be written as .csv files to ```ptss-benchmarks/```.
3. Train the ENC-SCO layer models by running ```python create_triple_vectors.py```. All models will be saved to the ```train_stats/``` directory.
4. Select the triples from each graph that participate in multiple predicates by running ```python find_multipredicate.py```.
5. The results for the clustering analysis can be obtained by running ```python reproduce_clusterability.py```.
6. The results for the edge classification task can be obtained by running ```python reproduce_classification.py```.
7. [OPTIONAL] T-SNE plots for all the triple embeddings can be produced by running ```python plot_tsne.py```.

## Acknowledgments

We are grateful to the authors of the LibKGE library for their best hyperparameter configuration files and pre-trained models:

```
@inproceedings{
  libkge,
  title="{L}ib{KGE} - {A} Knowledge Graph Embedding Library for Reproducible Research",
  author={Samuel Broscheit and Daniel Ruffinelli and Adrian Kochsiek and Patrick Betz and Rainer Gemulla},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations},
  year={2020},
  url={https://www.aclweb.org/anthology/2020.emnlp-demos.22},
  pages = "165--174",
}
```

We also heavily leverage the Siamese architectures found in SentBERT and rely their SentenceTransformers package code for our model training:

```
@inproceedings{reimers-2019-sentence-bert,
  title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
  author = "Reimers, Nils and Gurevych, Iryna",
  booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
  month = "11",
  year = "2019",
  publisher = "Association for Computational Linguistics",
  url = "https://arxiv.org/abs/1908.10084",
}
```

Finally, we are grateful to the authors of the original triple2vec paper for their communication and code sharing:

```
@misc{fionda2019triple2vec,
      title={Triple2Vec: Learning Triple Embeddings from Knowledge Graphs}, 
      author={Valeria Fionda and Giuseppe Pirró},
      year={2019},
      eprint={1905.11691},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```

