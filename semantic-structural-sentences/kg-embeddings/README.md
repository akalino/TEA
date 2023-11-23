# Knowledge Graph Embeddings

## TransE

For our knowledge graph embeddings, we rely on https://github.com/thunlp/OpenKE, please follow their install 
instructions (clone and compile) to build the necessary toolkit. Then
execute train_transe.py to train TransE embeddings. 

To extract the embeddings from
the model, first create a directory called data. 
Subsequently run build_node_space.py with the arguments -d 300 -m transe -s {dataset}.