import torch


class CharVocab:
    """
        Encoding and Decoding sentence to sequence of integer
    """

    def __init__(self, alphabet=None):

        if alphabet:
            with open(alphabet, 'r', encoding="utf-8") as file:
                self.alphabet = file.read()

        else:
            self.alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            self.alphabet += "abcdefghijklmnopqrstuvwxyz"
            self.alphabet += "0123456789"
            self.alphabet += '"'
            self.alphabet += "'`!^@#$%&.,?:;~-+*=_/\\|[]{}()<> "

        self.idx_to_char = self.alphabet

        pad, unk, bos, eos = '\0', '\1', '\2', '\3'
        special = [pad, unk, bos, eos]

        assert all(c not in self.idx_to_char for c in special)

        self.idx_to_char += "".join(special)

        self.char_to_idx = {self.idx_to_char[i]: i for i in range(len(self.idx_to_char))}

        self.bos_idx = self.char_to_idx[bos]
        self.eos_idx = self.char_to_idx[eos]
        self.pad_idx = self.char_to_idx[pad]
        self.unk_idx = self.char_to_idx[unk]

        self.size = len(self.char_to_idx)


    def save(self, path):
        with open(path, "w", encoding="utf-8") as file:
            file.write(self.alphabet)


    def decode_batch(self, batch):
        """
            Decoding batch [batch_size, max_len] to lines
        """
        return (
            "".join(self.idx_to_char[idx]
                    if idx < self.size
                    and idx not in [
                        self.bos_idx,
                        self.eos_idx,
                        self.pad_idx,
                        self.unk_idx
                    ] else ""
                for idx in x
            )
            for x in batch
        )


    def encode_batch(self, lines, device, max_len=None):
        """
            Encoding lines of sentence to tensor [batch_size, max_len + 1]
        """
        max_len = max_len if max_len else max(map(len, lines))

        idxs = [
            [
                self.char_to_idx[c]
                    if c in self.char_to_idx
                    else self.unk_idx
                for c in line[:max_len]
            ]
            for line in lines
        ]

        bos_idxs = [[self.bos_idx] + idx + [self.pad_idx] * (max_len + 1 - len(idx)) for idx in idxs]
        idxs_eos = [idx + [self.eos_idx] + [self.pad_idx] * (max_len + 1 - len(idx)) for idx in idxs]

        bos_idxs = torch.as_tensor(bos_idxs, dtype=torch.long).to(device)
        idxs_eos = torch.as_tensor(idxs_eos, dtype=torch.long).to(device)

        return bos_idxs, idxs_eos


