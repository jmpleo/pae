
import os

import itertools

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from pae.model import AAE, DAE, VAE, reparameterize
from pae.vocab import CharVocab
from pae.dataset import PIIDataset

from utils import logging, dump


class SessionPAE:
    def __init__(self,
        model_path,
        pii_load,
        sigma_min,
        sigma_max,
        sigmas_n,
        #min_len,
        #max_len,
        save_dir,
        log_file,
        batch_size=1000,
        alphabet=None,
        stdout=False,
        wordlist=None,
        cuda=False
    ):

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigmas_n = sigmas_n

        #self.min_len = min_len
        #self.max_len = max_len

        self.samples_file = os.path.join(save_dir, "wordlist.txt")
        self.log_file = log_file

        self.stdout = stdout
        self.wordlist = wordlist

        ############################################################################
        ## vocab setup
        ############################################################################
        self.vocab = CharVocab() #alphabet) ## NOTICE alphabet inference using be deprecated

        ############################################################################
        ## device setup
        ############################################################################
        self.device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")

        logging(log_file, f"# powered on {self.device}")

        arch = {
            "aae": AAE,
            "dae": DAE,
            "vae": VAE
        }

        ckpt = torch.load(model_path, map_location=self.device)

        self.model = arch[ckpt['args'].model_type](self.vocab, ckpt['args']).to(self.device)

        self.model.load_state_dict(ckpt['model'])
        self.model.flatten()

        n_param = sum(x.data.nelement() for x in self.model.parameters())

        logging(
            log_file,
            f"# model args:\n{ckpt['args']}\n"
            f"# model {ckpt['args'].model_type} parameters: {n_param}"
        )

        self.max_len = ckpt['args'].max_len

        ############################################################################
        ## check param
        ############################################################################
        # if self.max_len > model_max_len:

        #     logging(
        #         self.log_file,
        #         f"# WARNING! max_model_len ({model_max_len}) < max_len ({self.max_len})"
        #     )


        # if self.max_len < self.min_len:

        #     logging(
        #         self.log_file,
        #         f"# WARNING! max_len ({self.max_len}) < min_len ({self.min_len}), "
        #         f"# so using min_len =max_len ({self.max_len})"
        #     )

        #     self.min_len = self.max_len

        ############################################################################
        ## pii proccess
        ############################################################################
        self.batch_size = batch_size

        self.pii_dataloader = DataLoader(
            batch_size=self.batch_size,
            dataset=PIIDataset(path=pii_load, vocab=self.vocab),
            shuffle=True
        )

        logging(
            log_file,
            f"# pii idents {len(self.pii_dataloader.dataset)}"
        )

        self.limit_passwords = (
            len(self.pii_dataloader.dataset) * self.sigmas_n #* (self.max_len + 1 - self.min_len)
        )

    def run(self):

        ############################################################################
        ## fisrt popular wordlists
        ############################################################################
        if self.wordlist:
            logging(
                self.log_file,
                f"# dump wordlist {self.wordlist}"
            )

            with open(self.wordlist, 'r', encoding="utf-8") as wordlist:
                dump((word[:-1] for word in wordlist), self.samples_file, self.stdout)


        logging(
            self.log_file,
            f"# generation of {self.limit_passwords} passwords has been started"
        )

        sigmas = torch.linspace(self.sigma_min, self.sigma_max, self.sigmas_n).to(self.device)

        with tqdm(
                total=self.limit_passwords, desc='passwords', unit='password', leave=False,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} - {rate_fmt}'
            ) as pbar, tqdm(
                total=len(self.pii_dataloader), desc='batches', unit='batch', leave=False,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'
            ) as bbar:

            #for i, passwords_batch in enumerate(tqdm(self.pii_dataloader, desc="batches", unit="batch")):
            for i, passwords_batch in enumerate(self.pii_dataloader):

                inputs, _ = self.vocab.encode_batch(device=self.device, lines=passwords_batch)
                inputs = inputs.t().contiguous()

                # [batch_size, dim_z]
                pii_latents = reparameterize(*self.model.encode(inputs))

                # TODO: deprecate len choise
                # for length in range(self.min_len, self.max_len + 1): #, desc="Generating lengths", unit="length", leave=False):

                #for sigma in tqdm(torch.split(sigmas, self.batch_size // len(pii_latents)), desc="sigmas", unit="sigma", leave=False):
                for sigma in torch.split(sigmas, self.batch_size // len(pii_latents)):

                    same_zs = torch.normal(
                        pii_latents.unsqueeze(0), sigma.view(-1, 1, 1)
                    ).view(-1, self.model.opt.dim_z).to(self.device)

                    seq_batch = self.model.generate(same_zs, self.max_len)

                    gen_batch = self.vocab.decode_batch(seq_batch)

                    passwords = (password for password in gen_batch)

                    dump(passwords, self.samples_file, self.stdout)

                    pbar.n += same_zs.size(0)
                    pbar.refresh()

                    #logging(
                    #    self.log_file,
                    #    f"| batch {i + 1:2d}/{len(self.pii_dataloader)} "
                    #    f"| len {length:2d} "
                    #)

                bbar.n = i + 1
                bbar.refresh()

        logging(
            self.log_file,
            f"# total passwords {self.limit_passwords} generated successful"
        )

