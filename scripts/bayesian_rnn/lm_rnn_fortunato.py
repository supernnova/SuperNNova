import os
import numpy as np
import random
import argparse
import time
import math
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path

from supernnova.modules.bayesian_layers import (
    BayesEmbedding,
    BayesLSTM,
    BayesLinear,
    Prior,
)

from data import Corpus, batchify, get_batch, repackage_hidden


########################################################
# Code adapted from:
# https://github.com/pytorch/examples/tree/master/word_language_model
########################################################


def evaluate(model, criterion, corpus, data_source, eval_batch_size):

    model.eval()
    total_loss = 0.0
    total_words = 0.0
    total_entropy = 0.0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i, args.bptt)
            output, hidden = model(data, hidden, mean_field_inference=True)
            output_flat = output.view(-1, ntokens)

            num_words = output_flat.shape[0]
            pred_proba = nn.functional.softmax(output_flat, dim=-1)
            loss = len(data) * criterion(output_flat, targets).item() / num_words
            entropy = -(pred_proba * pred_proba.log()).sum(1).sum(0).item()

            total_words += num_words
            total_entropy += entropy
            total_loss += loss

            hidden = repackage_hidden(hidden)

    return total_loss / (len(data_source) - 1), total_entropy / total_words


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(LanguageModel, self).__init__()

        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        prior = Prior()

        init_recurrent = {
            "mu_lower": -0.05,
            "mu_upper": 0.05,
            "rho_lower": math.log(math.exp(prior.sigma_mix / 4.0) - 1.0),
            "rho_upper": math.log(math.exp(prior.sigma_mix / 2.0) - 1.0),
        }

        init_non_recurrent = {
            "mu_lower": -0.05,
            "mu_upper": 0.05,
            "rho_lower": math.log(math.exp(prior.sigma_mix / 2.0) - 1.0),
            "rho_upper": math.log(math.exp(prior.sigma_mix / 1.0) - 1.0),
        }

        # Layers
        self.encoder = BayesEmbedding(
            vocab_size, hidden_size, prior, **init_non_recurrent
        )
        self.bayeslstm = BayesLSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            prior=prior,
            **init_recurrent,
        )
        self.linear = BayesLinear(hidden_size, vocab_size, prior, **init_non_recurrent)

        self.kl = None

    def forward(self, x, hidden, mean_field_inference=False):

        embedding = self.encoder(x, mean_field_inference=mean_field_inference)
        out, hidden = self.bayeslstm(
            embedding, hidden, mean_field_inference=mean_field_inference
        )
        logits = self.linear(out, mean_field_inference=mean_field_inference)

        self.kl = self.encoder.kl + self.linear.kl + self.bayeslstm.kl

        return logits, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (
            weight.new_zeros(self.num_layers, batch_size, self.hidden_size),
            weight.new_zeros(self.num_layers, batch_size, self.hidden_size),
        )


def train_epoch(model, criterion, corpus, train_data, epoch, lr):

    model.train()
    total_likelihood_loss = 0.0
    total_kl_loss = 0.0

    assert criterion.reduction == "sum"

    num_batches = train_data.size(0) // args.bptt

    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    desc = f"Training epoch {epoch}"
    for batch, i in enumerate(
        tqdm(range(0, train_data.size(0) - 1, args.bptt), desc=desc)
    ):
        data, targets = get_batch(train_data, i, args.bptt)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()

        output, hidden = model(data, hidden)

        # Compute losses
        likelihood_loss = criterion(output.view(-1, ntokens), targets)
        kl_loss = model.kl / (num_batches)  # scaled by the number of batches

        # Further scale by the batch size to avoid large gradients
        likelihood_loss = likelihood_loss / args.batch_size
        kl_loss = kl_loss / args.batch_size

        loss = likelihood_loss + kl_loss
        loss.backward()

        total_likelihood_loss += likelihood_loss.detach()
        total_kl_loss += kl_loss.detach()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        for p in model.parameters():
            if p.grad is not None:
                d_p = p.grad.data
                p.data.add_(-lr, d_p)

        if batch % args.log_interval == 0 and batch > 0:
            cur_likelihood_loss = total_likelihood_loss / (
                args.log_interval * args.bptt
            )
            cur_kl_loss = total_kl_loss / args.log_interval
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | "
                "loss {:5.2f} | ppl {:8.2f} | KL {:8.2f}".format(
                    epoch,
                    batch,
                    len(train_data) // args.bptt,
                    lr,
                    elapsed * 1000 / args.log_interval,
                    cur_likelihood_loss,
                    math.exp(cur_likelihood_loss),
                    cur_kl_loss,
                )
            )
            total_likelihood_loss = 0
            total_kl_loss = 0
            start_time = time.time()


def run(args):

    device = torch.device("cuda" if args.cuda else "cpu")

    dir_path = os.path.dirname(os.path.realpath(__file__))
    debug_msg = (
        f"\n\nFirst download the PTB dataset and dump it to {dir_path}"
        "\nSee: https://github.com/townie/PTB-dataset-from-Tomas-Mikolov-s-webpage/tree/master/data"
    )
    for f in ["ptb.train.txt", "ptb.test.txt", "ptb.valid.txt"]:
        assert (Path(dir_path) / f).exists(), debug_msg

    eval_batch_size = 20
    corpus = Corpus(dir_path)
    train_data = batchify(corpus.train, args.batch_size, device)
    val_data = batchify(corpus.valid, eval_batch_size, device)
    test_data = batchify(corpus.test, eval_batch_size, device)
    rev_test_data = batchify(
        corpus.test[
            torch.arange(corpus.test.shape[0] - 1, -1, step=-1).to(corpus.test.device)
        ],
        eval_batch_size,
        device,
    )

    ntokens = len(corpus.dictionary)
    model = LanguageModel(ntokens, args.hidden_size, args.num_layers).to(device)

    criterion = nn.CrossEntropyLoss(reduction="sum")

    # Loop over epochs.
    lr = args.lr
    best_val_loss = 1e9

    # At any point you can hit Ctrl + C to break out of training early.
    for epoch in range(0, args.epochs):
        epoch_start_time = time.time()
        train_epoch(model, criterion, corpus, train_data, epoch, lr)
        val_loss, val_entropy = evaluate(
            model, criterion, corpus, val_data, eval_batch_size
        )
        print("-" * 89)
        print(
            "| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | "
            "valid ppl {:8.2f} | valid entropy {:8.2f}".format(
                epoch,
                (time.time() - epoch_start_time),
                val_loss,
                math.exp(val_loss),
                val_entropy,
            )
        )
        print("-" * 89)
        # Learning rate annealing
        if epoch >= 19:
            lr = lr * args.lr_decay

        print("=" * 89)
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), "model.pt")

    # Run on test data.
    test_loss, test_entropy = evaluate(
        model, criterion, corpus, test_data, eval_batch_size
    )
    _, rev_test_entropy = evaluate(
        model, criterion, corpus, rev_test_data, eval_batch_size
    )
    print("=" * 89)
    print(
        "| End of training | test loss {:5.2f} | test ppl {:8.2f} |"
        " test entropy {:8.2f} | delta entropy {:8.2f}".format(
            test_loss,
            math.exp(test_loss),
            test_entropy,
            rev_test_entropy - test_entropy,
        )
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="PyTorch PTB Bayesian RNN language model"
    )

    parser.add_argument("--hidden_size", type=int, default=650)
    parser.add_argument("--num_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--lr", type=float, default=1, help="initial learning rate")
    parser.add_argument("--lr_decay", type=float, default=0.9)
    parser.add_argument("--clip", type=float, default=5, help="gradient clipping")
    parser.add_argument("--epochs", type=int, default=70, help="upper epoch limit")
    parser.add_argument("--batch_size", type=int, default=20, help="batch size")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout")
    parser.add_argument("--bptt", type=int, default=35, help="sequence length")
    parser.add_argument("--seed", type=int, default=1111, help="random seed")
    parser.add_argument("--cuda", action="store_true", help="use CUDA")
    parser.add_argument("--log-interval", type=int, default=200, help="report interval")
    parser.add_argument("--save", type=str, default="model.pt", help="save path")

    args = parser.parse_args()

    # Set the random seed manually for reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if torch.cuda.is_available():
        if not args.cuda:
            print(
                "WARNING: You have a CUDA device, so you should probably run with --cuda"
            )

    run(args)
