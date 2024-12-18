import math
import numpy as np

import torch
import torch.nn.functional as F

from torch import nn
from torch import optim

from .noise import noisy


def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)


def log_prob(z, mu, logvar):
    var = torch.exp(logvar)
    logp = - (z-mu)**2 / (2*var) - torch.log(2*np.pi*var) / 2
    return logp.sum(dim=1)


def loss_kl(mu, logvar):
    return -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp()
        ) / len(mu)


class TextModel(nn.Module):
    def __init__(self, vocab, opt, initrange=0.1):
        super().__init__()
        self.vocab = vocab
        self.opt = opt
        self.emb = nn.Embedding(vocab.size, opt.dim_emb)
        self.out = nn.Linear(opt.dim_h, vocab.size)

        self.emb.weight.data.uniform_(-initrange, initrange)
        self.out.bias.data.zero_()
        self.out.weight.data.uniform_(-initrange, initrange)


class DAE(TextModel):
    def __init__(self, vocab, opt):
        super().__init__(vocab, opt)
        self.drop = nn.Dropout(opt.dropout)
        self.enc = nn.LSTM(
            input_size=opt.dim_emb,
            hidden_size=opt.dim_h,
            num_layers=opt.nlayers,
            dropout=opt.dropout if opt.nlayers > 1 else 0,
            bidirectional=True
        )
        self.dec = nn.LSTM(
            input_size=opt.dim_emb,
            hidden_size=opt.dim_h,
            num_layers=opt.nlayers,
            dropout=opt.dropout if opt.nlayers > 1 else 0
        )
        self.mu = nn.Linear(2*opt.dim_h, opt.dim_z)
        self.logvar = nn.Linear(2*opt.dim_h, opt.dim_z)
        self.emb_z = nn.Linear(opt.dim_z, opt.dim_emb)
        self.optim = optim.Adam(
            self.parameters(),
            lr=opt.lr,
            betas=(opt.b1, opt.b2)
        )


    def flatten(self):
        self.enc.flatten_parameters()
        self.dec.flatten_parameters()


    def encode(self, inp):
        # emb: [seq_len, batch_size, emb_size]
        emb = self.drop(self.emb(inp))

        # h: [2, batch_size, hidden_size]
        _, (h, _) = self.enc(emb)

        # h: [batch_size, hidden_size * 2]
        h = torch.cat([h[-2], h[-1]], dim=1)

        # mu: [batch_size, dim_z]
        # logvar: [batch_size, dim_z]
        return self.mu(h), self.logvar(h)


    def decode(self, z, inp, hid=None):
        # z: [batch_size, dim_z]
        # inp: [seq_len, batch_size]
        # emb: [seq_len, batch_size, emb_size]
        emb = self.drop(self.emb(inp)) + self.emb_z(z)

        # dec_seq: [seq_len, batch_size, hidden_size]
        # hid: [1, batch_size, hidden_size]
        dec_seq, hid = self.dec(emb, hid)
        dec_seq = self.drop(dec_seq)

        # out: [seq_len * batch_size, vocab_size]
        out = self.out(dec_seq.view(-1, dec_seq.size(-1)))

        # logits: [seq_len, batch_size, vocab_size]
        logits = out.view(dec_seq.size(0), dec_seq.size(1), -1)
        return logits, hid


    def generate(self, z, max_len, alg="greedy"):

        #assert alg in ['greedy']
        
        # z: [batch_size, dim_z]
        passwds = torch.full(
            # [batch_size, max_len + 1]
            size=(len(z), max_len + 1),
            fill_value=self.vocab.pad_idx,
            dtype=torch.long,
            device=z.device
        )

        passwds[:,0] = torch.full(
            size=(len(z),),
            fill_value=self.vocab.bos_idx,
            dtype=torch.long,
            device=z.device
        )

        hid = None

        for i in range(max_len):

            # passwds[:,i]: [batch_size, 1] -> inp: [1, batch_size]
            # logits: [1, batch_size, vocab_size]
            # hid: [1, batch_size, hidden_size]
            logits, hid = self.decode(z, passwds[:,i].view(1, -1), hid)

            passwds[:,i + 1] = logits.argmax(dim=-1)

        # [batch_size, max_len]
        return passwds


    def forward(self, inp, is_train=False):
        inp = noisy(self.vocab, inp, *self.opt.noise) if is_train else inp

        # mu: [batch_size, dim_z]
        # logvar: [batch_size, dim_z]
        mu, logvar = self.encode(inp)

        # z: [batch_size, dim_z]
        z = reparameterize(mu, logvar)

        # logits: [seq_len, batch_size, vocab_size]
        logits, _ = self.decode(z, inp)
        return mu, logvar, z, logits


    def loss_rec(self, logits, targets):
        """
            reconstruction loss using binary cross entropy
            logits: [seq_len, batch_size, vocab_size]
            targets: [seq_len, batch_size]

            output: [batch_size]
        """
        loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.vocab.pad_idx,
                reduction='none'
            ).view(targets.size())

        return loss.sum(dim=0)


    def loss(self, losses):
        return losses['rec']


    def autoenc(self, inps, targets, is_train=False):
        _, _, _, logits = self(inps, is_train)
        return { 'rec': self.loss_rec(logits, targets).mean() }


    def step(self, losses):
        self.optim.zero_grad()
        losses['loss'].backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        #nn.utils.clip_grad_norm_(self.parameters(), clip)
        self.optim.step()


    def neg_log_loke_is(self, inps, targets, m):
        """compute negative log-likelihood by importance sampling:
           p(x;theta) = E_{q(z|x;phi)}[p(z)p(x|z;theta)/q(z|x;phi)]
        """
        mu, logvar = self.encode(inps)
        tmp = []
        for _ in range(m):
            z = reparameterize(mu, logvar)
            logits, _ = self.decode(z, inps)
            v = log_prob(
                    z,
                    torch.zeros_like(z),
                    torch.zeros_like(z)
                ) - self.loss_rec(logits, targets
                ) - log_prob(z, mu, logvar)
            tmp.append(v.unsqueeze(-1))
        ll_is = torch.logsumexp(torch.cat(tmp, 1), 1) - np.log(m)
        return -ll_is


class VAE(DAE):
    """Variational Auto-Encoder"""

    def __init__(self, vocab, opt):
        super().__init__(vocab, opt)

    def loss(self, losses):
        return losses['rec'] + self.opt.lambda_kl * losses['kl']

    def autoenc(self, inps, targets, is_train=False):
        mu, logvar, _, logits = self(inps, is_train)
        return {
            'rec': self.loss_rec(logits, targets).mean(),
            'kl': loss_kl(mu, logvar)
        }


class AAE(DAE):
    """Adversarial Auto-Encoder"""

    def __init__(self, vocab, opt):
        super().__init__(vocab, opt)
        self.discrim = nn.Sequential(
                nn.Linear(opt.dim_z, opt.dim_d),
                nn.ReLU(),
                nn.Linear(opt.dim_d, 1),
                nn.Sigmoid()
            )
        self.discrim_optim = optim.Adam(
                self.discrim.parameters(),
                lr=opt.lr,
                betas=(opt.b1, opt.b2)
            )


    def loss_adv(self, z):
        zn = torch.randn_like(z)
        zeros = torch.zeros(len(z), 1, device=z.device)
        ones = torch.ones(len(z), 1, device=z.device)
        loss_d = F.binary_cross_entropy(
                self.discrim(z.detach()),
                zeros
            ) + F.binary_cross_entropy(
                self.discrim(zn),
                ones
            )
        loss_g = F.binary_cross_entropy(
                self.discrim(z),
                ones
            )
        return loss_d, loss_g


    def loss(self, losses):
        return losses['rec'
            ] + self.opt.lambda_adv * losses['adv'
            ] + self.opt.lambda_p * losses['|lvar|']


    def autoenc(self, inps, targets, is_train=False):
        _, logvar, z, logits = self(inps, is_train)
        loss_d, adv = self.loss_adv(z)
        return {
                'rec': self.loss_rec(logits, targets).mean(),
                'adv': adv,
                '|lvar|': logvar.abs().sum(dim=1).mean(),
                'loss_d': loss_d
            }


    def step(self, losses):
        super().step(losses)

        self.discrim_optim.zero_grad()
        losses['loss_d'].backward()
        self.discrim_optim.step()

