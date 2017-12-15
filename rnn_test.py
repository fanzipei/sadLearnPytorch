# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

USE_CUDA = True


class GRUPredictor(nn.Module):
    def __init__(self, embed_size, hidden_size, dict_size):
        super(GRUPredictor, self).__init__()

        self.embedding = nn.Embedding(dict_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, dict_size)

    def forward(self, input_loc, last_hidden):
        loc_embedded = self.embedding(input_loc).view(1, 1, -1)
        output, hidden = self.gru(loc_embedded, last_hidden)

        output = output.squeeze(0)
        output = F.log_softmax(self.out(output))

        return output, hidden

