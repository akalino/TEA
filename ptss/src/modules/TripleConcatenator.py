import torch
import torch.nn as nn


class TripleConcatenator(nn.Module):

    def __init__(self, _model, _concat):
        super().__init__()
        self.model = _model
        self.concat_method = _concat
        self.entity_tensor, self.rel_tensor = self.fetch_embeddings(self.model)
        self.entity_embedding, self.ent_dim = self.create_emb_layer(self.entity_tensor, False)
        self.rel_embedding, self.rel_dim = self.create_emb_layer(self.rel_tensor, False)

    def fetch_embeddings(self, _path):
        mod = torch.load(_path, map_location=torch.device('cuda:0'))
        try:
            _ents = mod['model'][0]['_entity_embedder.embeddings.weight'].cpu()
            _rels = mod['model'][0]['_relation_embedder.embeddings.weight'].cpu()
        except KeyError:
            _ents = mod['model'][0]['_entity_embedder._embeddings.weight'].cpu()
            _rels = mod['model'][0]['_relation_embedder._embeddings.weight'].cpu()
        return _ents, _rels

    def represent_triple(self, _h, _r, _t):
        if self.concat_method == 'concat':
            _triple = torch.cat([self.entity_embedding(_h), self.rel_embedding(_r), self.entity_embedding(_t)], dim=0)
        elif self.concat_method == 'ht':
            _triple = torch.cat([self.entity_embedding(_h), self.entity_embedding(_t)], dim=0)
        elif self.concat_method == 'rel':
            _triple = self.rel_embedding(_r)
        elif self.concat_method == 'add':
            _triple = self.entity_embedding(_h) + self.rel_embedding(_r) + self.entity_embedding(_t)
        _norm_triple = nn.functional.normalize(_triple, p=2, dim=0)
        return _norm_triple

    @staticmethod
    def create_emb_layer(weights_matrix, trainable=False):
        num_embeddings, embedding_dim = weights_matrix.size()
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight': weights_matrix})
        if trainable:
            emb_layer.weight.requires_grad = True
        else:
            emb_layer.weight.requires_grad = False
        return emb_layer, embedding_dim
