# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

USE_CUDA = True


class Seq2seqVAE(nn.Module):
    def __init__(self, embed_size, hidden_size, dict_size, n_layers=1, dropout_p=0.1):
        super(Seq2seqVAE, self).__init__()

        self.embed = nn.Embedding(dict_size, embed_size)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(embed_size, hidden_size)
