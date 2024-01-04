import torch
import torch.nn as nn


class TripleDistilBert(nn.module):

    def __init__(self, _model, _concat):
        super().__init__()
