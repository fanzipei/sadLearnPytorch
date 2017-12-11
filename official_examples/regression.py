# -*- coding: utf-8 -*-

from itertools import count

import torch
import torch.autograd
import torch.nn.functional as F
from torch.autograd import Variable

POLY_DEGREE = 4
W_target = torch.randn(POLY_DEGREE, 1) * 5
b_target = torch.randn(1) * 5


def make_features(x):
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, POLY_DEGREE + 1)], 1)


def f(x):
    return x.mm(W_target) + b_target[0]


def poly_desc(W, b):
    result = 'y = '
    for i, w in enumerate(W):
        result += '{:+.2f} x^{} '.format(w, len(W) - i)
    result += '{:+.2f}'.format(b[0])
    return result


def get_batch(batch_size=32):
    random = torch.randn(batch_size)
    x = make_features(random)
    y = f(x)
    return Variable(x), Variable(y)


fc = torch.nn.Linear(W_target.size(0), 1)

for batch_idx in count(1):
    batch_x, batch_y = get_batch()
    fc.zero_grad()
    output = F.smooth_l1_loss(fc(batch_x), batch_y)
    loss = output.data[0]

    output.backward()

    for param in fc.parameters():
        param.data.add_(-0.1 * param.grad.data)

    if loss < 1e-3:
        break

print('Loss: {:.6f} after {} batches'.format(loss, batch_idx))
print('==> learned function:\t' + poly_desc(fc.weight.data.view(-1), fc.bias.data))
print('==> Actual function:\t' + poly_desc(W_target.view(-1), b_target))
