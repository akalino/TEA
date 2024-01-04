import torch
import torch.nn as nn


class WordEmbedding(nn.Module):
    def __init__(self,
                 _tokenizer: None,
                 _word_embedding_dim: int,
                 _vocab_size: int,
                 _max_sent_len: int):
        super(Embedding, self).__init__()
        self.tokenizer = _tokenizer
        self.word_embedding_dim = _word_embedding_dim
        self.vocab_size = _vocab_size
        self.max_sent_len = _max_sent_len
        self.word_embed = nn.Embedding(self.vocab_size, self.word_embedding_dim)
        self.pos_embed = nn.Embedding(self.max_sent_len, self.word_embedding_dim)
        self.norm = nn.LayerNorm(self.word_embedding_dim)
        self.dropout = nn.Dropout(0.1, inplace=False)

    def forward(self, features):
        seq_lens = features['sentence_lengths']
        poss = [torch.arange(x) for x in seq_lens]
        tok_es = []
        for j in range(features['input_ids'].shape(0)):
            tok_es.append(self.dropout(self.norm(self.tok_embed(features['input_ids'][j]) + self.pos_embed(poss[j]))))
        cls_tokens = None
        token_embeddings = torch.stack(tok_es)
        features.update({'token_embeddings': token_embeddings,
                         'cls_token_embeddings': cls_tokens,
                         'attention_mask': features['attention_mask']})
        return features

    def tokenize(self, texts: List[str]):
        tokenized_texts = [self.tokenizer.tokenize(text) for text in texts]
        sentence_lengths = [len(tokens) for tokens in tokenized_texts]
        max_len = max(sentence_lengths)

        input_ids = []
        attention_masks = []
        for tokens in tokenized_texts:
            padding = [0] * (max_len - len(tokens))
            input_ids.append(tokens + padding)
            attention_masks.append([1] * len(tokens) + padding)

        output = {'input_ids': torch.tensor(input_ids, dtype=torch.long),
                  'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
                  'sentence_lengths': torch.tensor(sentence_lengths, dtype=torch.long)}

        return output
