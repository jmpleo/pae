import argparse
import os
import math
import collections
import time

from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from torch import Tensor

import torch.nn as nn
import torch.nn.functional as F
import torch

from pae.vocab import CharVocab
from pae.dataset import PasswordDataset
from pae.model import AAE, DAE, VAE
from pae.meter import AverageMeter
from utils import set_seed, logging

parser = argparse.ArgumentParser()

# path param
parser.add_argument(
    "--train",
    metavar="FILE",
    required=True,
    help="path to leak passwords for train"
)

parser.add_argument(
    "--valid",
    metavar="FILE",
    required=True,
    help="path to leak passwords for valid"
)

parser.add_argument(
    "--alphabet",
    metavar="FILE",
    required=False,
    help="path to alphabet file"
)

parser.add_argument(
    "--save_dir",
    default="checkpoints",
    metavar="DIR",
    help="directory to save checkpoints and outputs",
)

parser.add_argument(
    "--load_model",
    default="",
    metavar="FILE",
    help="path to load checkpoint if specified",
)

# model param
parser.add_argument(
    "--model_type",
    default="aae",
    metavar="M",
    choices=["dae", "vae", "aae"],
    help="which model to learn",
)

parser.add_argument(
    "--max_len",
    type=int,
    required=True,
    metavar="D",
    help="max length of passwords"
)

parser.add_argument(
    "--dim_z",
    type=int,
    default=128,
    metavar="D",
    help="dimension of latent variable z"
)

parser.add_argument(
    "--dim_emb",
    type=int,
    default=64,
    metavar="D",
    help="dimension of word embedding"
)

parser.add_argument(
    "--dim_h",
    type=int,
    default=256,
    metavar="D",
    help="dimension of hidden state per layer",
)

parser.add_argument(
    "--nlayers",
    type=int,
    default=1,
    metavar="N",
    help="number of LSTM layers"
)

parser.add_argument(
    "--dim_d",
    type=int,
    default=512,
    metavar="D",
    help="dimension of hidden state in AAE discriminator",
)

# train param
parser.add_argument(
    "--lambda_kl",
    type=float,
    default=0,
    metavar="R",
    help="weight for kl term in VAE"
)

parser.add_argument(
    "--lambda_adv",
    type=float,
    default=1,
    metavar="R",
    help="weight for adversarial loss in AAE",
)

parser.add_argument(
    "--lambda_p",
    type=float,
    default=0,
    metavar="R",
    help="weight for L1 penalty on posterior log-variance",
)

parser.add_argument(
    "--noise",
    default="0.2,0.1,0",
    metavar="P,P,K",
    help="char drop prob, substitute prob, max char shuffle distance",
)

parser.add_argument(
    "--dropout",
    type=float,
    default=0.5,
    metavar="DROP",
    help="dropout probability (0 = no dropout)",
)

parser.add_argument(
    "--lr",
    type=float,
    default=0.0005,
    metavar="LR",
    help="adam learning rate"
)

parser.add_argument(
    "--b1",
    type=float,
    default=0.5,
    metavar="LR",
    help="adam decay rates - b1"
)

parser.add_argument(
    "--b2",
    type=float,
    default=0.999,
    metavar="LR",
    help="adam decay rates - b2"
)

parser.add_argument(
    "--epochs",
    type=int,
    default=50,
    metavar="N",
    help="number of training epochs"
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=256,
    metavar="N",
    help="batch size"
)

parser.add_argument(
    "--log_interval",
    type=int,
    default=100,
    metavar="N",
    help="report interval"
)

parser.add_argument(
    "--no_cuda",
    action="store_true",
    help="disable CUDA"
)


def evaluate(model, dataloader: DataLoader):
    """
    compute model metrics
    """
    model.eval()
    meters = collections.defaultdict(AverageMeter)
    with torch.no_grad():
        for inputs, targets in dataloader:

            # batch second
            inputs = inputs.t().contiguous()
            targets = targets.t().contiguous()

            losses = model.autoenc(inputs, targets)

            for k, v in losses.items():
                meters[k].update(v.item(), inputs.size(1))

    loss = model.loss({k: meter.avg for k, meter in meters.items()})
    meters["loss"].update(loss)
    return meters


def main(args):
    """
    Main function
    """

    ############################################################################
    ## parse noise
    ############################################################################
    parse_args.noise = [float(x) for x in parse_args.noise.split(",")]

    ############################################################################
    ## path init
    ############################################################################
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    log_file = os.path.join(args.save_dir, "log.txt")
    vocab_file = os.path.join(args.save_dir, "vocab.alphabet")
    model_file = os.path.join(args.save_dir, "model.pt")

    logging(log_file, str(args))

    ############################################################################
    ## device setup
    ############################################################################
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    logging(log_file, f"# train on {device} device")

    ############################################################################
    ## vocab init
    ############################################################################
    vocab = CharVocab(args.alphabet)
    vocab.save(vocab_file)

    logging(log_file, f"# vocab save {vocab_file}")

    ############################################################################
    ## train data load
    ############################################################################
    train_dataloader = torch.utils.data.DataLoader(
        batch_size=args.batch_size,
        dataset=PasswordDataset(
            file=args.train,
            vocab=vocab,
            device=device,
            max_len=args.max_len
        )
    )

    logging(log_file, f"# train passwords {len(train_dataloader.dataset)}")

    ############################################################################
    ## valid data load
    ############################################################################
    valid_dataloader = torch.utils.data.DataLoader(
        batch_size=args.batch_size,
        dataset=PasswordDataset(
            file=args.valid,
            vocab=vocab,
            device=device,
            max_len=args.max_len
        ),
    )

    logging(log_file, f"# valid passwords {len(valid_dataloader.dataset)}")

    ############################################################################
    ## model init
    ############################################################################
    models = {
        "dae": DAE,
        "vae": VAE,
        "aae": AAE
    }

    model = models[args.model_type](vocab, args).to(device)

    if args.load_model:
        ckpt = torch.load(args.load_model, map_location=device)
        model = models[ckpt["args"].model_type](vocab, args).to(device)
        model.load_state_dict(ckpt["model"])
        model.flatten()

    n_param = sum(x.data.nelement() for x in model.parameters())

    logging(log_file, f"# model {args.model_type} parameters: {n_param}")

    ############################################################################
    ## train loop
    ############################################################################
    best_val_loss = None
    for epoch in range(args.epochs):

        logging(log_file, "-" * 80)
        log_output = ""

        start_time = time.time()

        model.train()
        meters = collections.defaultdict(AverageMeter)

        for i, (inputs, targets) in enumerate(train_dataloader):

            # batch second
            inputs = inputs.t().contiguous()
            targets = targets.t().contiguous()

            losses = model.autoenc(inputs, targets, is_train=True)

            losses["loss"] = model.loss(losses)

            model.step(losses)

            for k, v in losses.items():
                meters[k].update(v.item())

            if (i + 1) % args.log_interval == 0:
                log_output = (
                    f"| epoch {epoch + 1: 3d} "
                    f"| {i + 1: 5d}/{len(train_dataloader): 5d} batches |"
                )

                for k, meter in meters.items():
                    log_output += f" {k} {meter.avg: .2f},"
                    meter.clear()

                logging(log_file, log_output)

        valid_meters = evaluate(model, valid_dataloader)

        logging(log_file, "-" * 80)

        log_output = (
            f"| end of epoch {epoch + 1:3d}"
            f"| time {time.time() - start_time: 5.0f}s"
            f"| valid"
        )

        for k, meter in valid_meters.items():
            log_output += f" {k} {meter.avg:.2f},"

        if not best_val_loss or valid_meters["loss"].avg < best_val_loss:

            ckpt = {
                "args": args,
                "model": model.state_dict()
            }

            torch.save(ckpt, model_file)

            best_val_loss = valid_meters["loss"].avg

            log_output += " | saving model"

        logging(log_file, log_output)

    logging(log_file, "Done training")


if __name__ == "__main__":
    parse_args = parser.parse_args()
    main(parse_args)
