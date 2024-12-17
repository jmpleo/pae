import os
import re
from itertools import groupby

from torch.utils.data import DataLoader, Dataset


def cut_lines(lines, alphabet):
    """
        erase lines which contains symbols over alphabet
    """
    return (line for line in lines if all(c in alphabet for c in line[:-1]))


class PIIDataset(Dataset):
    def __init__(self, path: str, vocab):

        # top 20 passwords
        text = """
            123456 123456789 12345 qwerty password 12345678
            111111 123123 1234567890 1234567 qwerty123 000000
            1q2w3e aa12345678 abc123 password1 1234 qwertyuiop
            123321 password123
        """

        if path and (os.path.isfile(path) or os.path.isdir(path)):
            pii_files = [path] if os.path.isfile(path) else (
                os.path.join(root, file)
                for root, _, files in os.walk(path)
                    for file in files
            )

            text = ""
            for pii_file in pii_files:
                with open(pii_file, 'r', encoding="utf-8") as f:
                    text += f.read()


        self.passwords = list(cut_lines(
            alphabet=vocab.alphabet,
            lines=PIIDataset.split(PIIDataset.preprocess(text))
        ))

        # self.pii_idxs, _ = vocab.encode_batch(
        #     device=device,
        #     lines=list(cut_lines(
        #         alphabet=vocab.alphabet,
        #         lines=PIIDataset.split(PIIDataset.preprocess(text))
        #     ))
        # )


    @staticmethod
    def split(text):
        delims = " .,\n\t"
        return "".join(c
            if c not in delims
            else ' '
            for c in text
        ).split()


    @staticmethod
    def drop_domain_mail(text):
        """
            drop domain mail from any text
        """
        return re.sub(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            lambda m: m.group().split('@')[0],
            text
        )


    @staticmethod
    def preprocess(text):
        """
            pii preprocess
        """
        text = PIIDataset.drop_domain_mail(text)

        return text


    def __len__(self):
        return len(self.passwords)

    def __getitem__(self, idx):
        return self.passwords[idx]


class PasswordDataset(Dataset):
    """
        load passwords fixed max_len
    """
    def __init__(self, file: str, vocab):

        with open(file, 'r', encoding="utf8") as f:

            self.passwords = [
                line[:-1] 
                for line in f if line[:-1] and all(
                    c in vocab.alphabet for c in line[:-1]
                )
            ]
        

    def __len__(self):
        return len(self.passwords)

    def __getitem__(self, idx):
        return self.passwords[idx]
    
