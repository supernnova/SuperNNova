import os
from io import open
import numpy as np
from pathlib import Path

import torch
from torchvision import transforms
from torchvision import datasets


def format_mnist():

    dir_path = os.path.dirname(os.path.realpath(__file__))
    mnist_path = Path(dir_path) / "data/mnist"

    try:
        return np.load(mnist_path / "mnist.npz")

    except Exception:

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                mnist_path,
                train=True,
                download=True,
                transform=transforms.Compose([transforms.ToTensor()]),
            ),
            batch_size=128,
            shuffle=True,
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                mnist_path,
                train=False,
                transform=transforms.Compose([transforms.ToTensor()]),
            ),
            batch_size=128,
            shuffle=True,
        )

        list_X = []
        list_Y = []

        for (x, y) in train_loader:
            list_X.append(x.view(-1, 784).detach().cpu().numpy())
            list_Y.append(y.detach().cpu().numpy())

        X_train = np.concatenate(list_X, axis=0)
        Y_train = np.concatenate(list_Y, axis=0)

        list_X = []
        list_Y = []

        for (x, y) in test_loader:
            list_X.append(x.view(-1, 784).detach().cpu().numpy())
            list_Y.append(y.detach().cpu().numpy())

        X_test = np.concatenate(list_X, axis=0)
        Y_test = np.concatenate(list_Y, axis=0)

        d_data = {
            "X_train": X_train,
            "Y_train": Y_train,
            "X_test": X_test,
            "Y_test": Y_test,
        }

        np.savez(mnist_path / "mnist.npz", **d_data)

        return d_data


########################################################
# Code adapted from:
# https://github.com/pytorch/examples/tree/master/word_language_model
########################################################


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, "train.txt"))
        self.valid = self.tokenize(os.path.join(path, "valid.txt"))
        self.test = self.tokenize(os.path.join(path, "test.txt"))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, "r", encoding="utf8") as f:
            tokens = 0
            for line in f:
                words = line.split() + ["<eos>"]
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, "r", encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ["<eos>"]
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids


######################################################################
# Load data for PTB Language modelling
#
# Code adapted from:
# https://github.com/pytorch/examples/tree/master/word_language_model
######################################################################


def batchify(data, bsz, device):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len].view(-1)
    return data, target
