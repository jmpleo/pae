import numpy as np
import torch


def tok_shuffle(vocab, x, k):
    # slight shuffle such that |sigma[i]-i| <= k
    seq_len, batch_size = x.size(0), x.size(1)

    # base: [seq_len, batch_size]
    base = torch.arange(seq_len, dtype=torch.float).repeat(batch_size, 1).t()

    inc = (k+2) * torch.rand(x.size())
    inc[x == vocab.bos_idx] = 0     # do not shuffle the start sentence symbol
    inc[x == vocab.eos_idx] = k+1    # do not shuffle end paddings
    inc[x == vocab.pad_idx] = k+2  # do not shuffle end paddings
    _, sigma = (base + inc).sort(dim=0)

    return x[
        sigma,
        torch.arange(x.size(1))
    ]

def tok_drop(vocab, x, p):     # drop words with probability p
    x_ = []
    batch_size = x.size(1)
    for i in range(batch_size):
        toks = x[:, i].tolist()
        keep = np.random.rand(len(toks)) > p
        keep[0] = True  # do not drop the start sentence symbol
        sent = [tok for j, tok in enumerate(toks) if keep[j]]
        sent += [vocab.pad_idx] * (len(toks)-len(sent))
        x_.append(sent)

    # [seq_len, batch_size]
    return torch.LongTensor(x_).t().contiguous().to(x.device)
"""
def tok_blank(vocab, x, p):     # blank words with probability p
    blank = (torch.rand(x.size(), device=x.device) < p) & \
        (x != vocab.eos_ix) & (x != vocab.pad_ix)
    x_ = x.clone()
    x_[blank] = vocab.blank
    return x_
"""

def tok_substitute(vocab, x, p):     # substitute words with probability p
    x_ = x.clone()
    x_.random_(vocab.size)

    keep = (torch.rand(x.size(), device=x.device) > p
            ) | (x == vocab.bos_idx
            ) | (x == vocab.eos_idx
            ) | (x == vocab.pad_idx
            ) | (x_ == vocab.bos_idx
            ) | (x_ == vocab.eos_idx
            ) | (x_ == vocab.pad_idx)

    x_[keep] = x[keep]
    return x_

def noisy(vocab, x, drop_prob, sub_prob, shuffle_dist):
    if shuffle_dist > 0:
        x = tok_shuffle(vocab, x, shuffle_dist)
    if drop_prob > 0:
        x = tok_drop(vocab, x, drop_prob)
    #if blank_prob > 0:
    #    x = word_blank(vocab, x, blank_prob)
    if sub_prob > 0:
        x = tok_substitute(vocab, x, sub_prob)
    return x
