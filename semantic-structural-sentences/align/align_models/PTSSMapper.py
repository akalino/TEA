import torch
import torch.nn as nn

from .BaseModel import BaseModel


class PTSSMapper(BaseModel):
    def __init__(self, _sent_weights, _triple_weights,
                 _concat, _normalize_vecs, _beta, _tune):
        super().__init__()
        # torch.manual_seed(17)
        self.concat_method = _concat
        self.sent_weights = _sent_weights
        self.triple_weights = _triple_weights
        self.normalize_vecs = _normalize_vecs
        self.normalize_embeddings()
        self.beta = _beta
        self.update_embeddings = _tune

        self.sent_embedding, self.sent_dim = self.create_emb_layer(self.sent_weights, _tune)
        self.triple_embedding, self.triple_dim = self.create_emb_layer(self.triple_weights, _tune)
        self.kg_dim = self.triple_dim
        self.mapping = nn.Linear(self.kg_dim, self.sent_dim, bias=False)
        torch.nn.init.orthogonal_(self.mapping.weight)

    def forward(self, _n_in):
        triples = self.triple_embedding(_n_in)
        out = self.mapping(triples)
        out_norm = nn.functional.normalize(out, p=2, dim=1)
        return out_norm

    def prior_space(self, _n_in):
        triples = self.triple_embedding(_n_in)
        return triples

    def orthogonalize(self):
        if self.beta > 0:
            W = self.mapping.weight.data
            beta = self.beta
            W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W)))
