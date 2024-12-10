
import os

import itertools

import torch
from torch.utils.data import DataLoader

from pae.model import AAE, DAE, VAE, reparameterize
from pae.vocab import CharVocab
from pae.dataset import PIIDataset

from utils import logging, dump


class SessionPAE:
    def __init__(self,
        load_model,
        pii_load,
        similar_std,
        similar_sample_n,
        min_len,
        max_len,
        save_dir,
        log_file,
        batch_size=1000,
        alphabet=None,
        stdout=False,
        wordlist=None
    ):

        self.similar_std = similar_std
        self.similar_sample_n = similar_sample_n

        self.min_len = min_len
        self.max_len = max_len

        self.samples_file = os.path.join(save_dir, "wordlist.txt")
        self.log_file = log_file

        self.stdout = stdout
        self.wordlist = wordlist

        ############################################################################
        ## vocab setup
        ############################################################################
        self.vocab = CharVocab(alphabet)

        ############################################################################
        ## device setup
        ############################################################################
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        logging(log_file, f"# powered on {device}")

        arch = {
            "aae": AAE,
            "dae": DAE,
            "vae": VAE
        }

        ckpt = torch.load(load_model, map_location=device)

        self.model = arch[ckpt['args'].model_type](self.vocab, ckpt['args']).to(device)

        self.model.load_state_dict(ckpt['model'])
        self.model.flatten()

        n_param = sum(x.data.nelement() for x in self.model.parameters())

        logging(
            log_file,
            f"# model args:\n{ckpt['args']}\n"
            f"# model {ckpt['args'].model_type} parameters: {n_param}"
        )

        model_max_len = ckpt['args'].max_len

        ############################################################################
        ## check param
        ############################################################################
        if self.max_len > model_max_len:

            logging(
                self.log_file,
                f"# WARNING! max_model_len ({model_max_len}) < max_len ({self.max_len})"
            )


        if self.max_len < self.min_len:

            logging(
                self.log_file,
                f"# WARNING! max_len ({self.max_len}) < min_len ({self.min_len}), "
                f"# so using min_len =max_len ({self.max_len})"
            )

            self.min_len = self.max_len


        ############################################################################
        ## pii proccess
        ############################################################################
        self.pii_dataloader = DataLoader(
            batch_size=batch_size,
            shuffle=True,
            dataset=PIIDataset(
                path=pii_load,
                vocab=self.vocab,
                device=device
            )
        )

        logging(
            log_file,
            f"# pii idents {len(self.pii_dataloader.dataset)}"
        )

        self.limit_passwords = len(self.pii_dataloader.dataset)
        self.limit_passwords *= (self.max_len + 1 - self.min_len)
        self.limit_passwords *= self.similar_sample_n


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


        spaces = (
            reparameterize(*self.model.encode(inputs.t().contiguous()))
            for inputs in self.pii_dataloader
        )

        for i, space in enumerate(spaces):

            for length in range(self.min_len, self.max_len + 1):

                similar_vectors = torch.normal(space, self.similar_std).to(space.device)

                seq_batches = (
                    self.model.generate(similar_vectors, length)
                    for _ in range(self.similar_sample_n)
                )

                password_batches = (
                    self.vocab.decode_batch(seq_batch)
                    for seq_batch in seq_batches
                )

                passwords = (
                    password
                    for password_batch in password_batches
                        for password in password_batch
                )

                dump(passwords, self.samples_file, self.stdout)

                logging(
                    self.log_file,
                    f"| batch {i + 1:2d}/{len(self.pii_dataloader)} "
                    f"| len {length:2d} "
                )


        logging(
            self.log_file,
            f"# total passwords {self.limit_passwords} generated successful"
        )

