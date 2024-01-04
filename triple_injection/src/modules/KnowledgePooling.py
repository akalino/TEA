import torch
import torch.nn as nn

from torch import Tensor


class KnowledgePooling(nn.module):
    def __init__(self,
                 _transport_map: None,
                 _triple_embeddings: Tensor,
                 _n_neighbors: int,
                 _pooling_type: str):
        super(self, KnowledgePooling).__init__()
        self.mapper = _transport_map
        self.triple_embeddings = _triple_embeddings
        self.n_neighbors = _n_neighbors
        self.pooling_type = _pooling_type

    def forward(self, features: Dict[str, Tensor]):
        sentence_embeddings = features['sentence_embedding']
        projs = self.mapper.project(sentence_embeddings,
                                    method='conditional')
