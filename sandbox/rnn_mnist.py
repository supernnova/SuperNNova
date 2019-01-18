import math
import random
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from data import format_mnist

from supernnova.training.bayesian_rnn import BayesLinear, BayesLSTM, Prior
from supernnova.training.variational_rnn import (
    VariationalDropout,
    VariationalRecurrentDropout,
    WeightDrop,
)


class RNN(nn.Module):

    def __init__(
        self,
        model,
        dropout,
        input_size,
        hidden_size,
        num_layers,
        bidirectional,
        prior,
        mu_lower,
        mu_upper,
        rho_lower,
        rho_upper,
    ):
        super(RNN, self).__init__()

        # Params
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 10
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.prior = prior
        self.model = model
        self.dropout = dropout

        # Layers / nn objects
        if "bayes" in self.model:
            self.rnn_layer = BayesLSTM(
                input_size,
                self.hidden_size,
                self.prior,
                num_layers=self.num_layers,
                bidirectional=self.bidirectional,
                mu_lower=-mu_lower,
                mu_upper=mu_upper,
                rho_lower=rho_lower,
                rho_upper=rho_upper,
            )

            self.output_layer = BayesLinear(
                self.hidden_size,
                self.output_size,
                self.prior,
                mu_lower=-mu_lower,
                mu_upper=mu_upper,
                rho_lower=rho_lower,
                rho_upper=rho_upper,
            )
        elif "variational" in self.model:
            self.recurrent_dropout_layer = VariationalRecurrentDropout()
            self.output_dropout_layer = VariationalDropout()
            self.output_layer = torch.nn.Linear(self.hidden_size, self.output_size)

            self.rnn_layers = []
            bidirectional_factor = 2 if self.bidirectional is True else 1
            for i in range(self.num_layers):
                if i == 0:
                    input_size = self.input_size
                else:
                    previous_layer = getattr(self, f"rnn_layer{i - 1}")
                    input_size = (
                        previous_layer.module.hidden_size * bidirectional_factor
                    )

                # Create recurrent layer
                layer = nn.LSTM(
                    input_size,
                    self.hidden_size,
                    num_layers=1,
                    dropout=0.,  # no dropout: we later create a specific layer for that
                    bidirectional=self.bidirectional,
                )
                # Apply weight drop
                layer = WeightDrop(layer, ["weight_hh_l0"], dropout=self.dropout)
                # Set layer as attribute
                setattr(self, f"rnn_layer{i}", layer)
                self.rnn_layers.append(layer)

        else:
            self.rnn_layer = nn.LSTM(
                input_size,
                self.hidden_size,
                num_layers=1,
                dropout=0,
                bidirectional=self.bidirectional,
            )
            self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, mean_field_inference=False):
        if "bayes" in self.model:
            x, hidden = self.rnn_layer(x, mean_field_inference=mean_field_inference)

            x = x.mean(0)

            # Final projection layer
            output = self.output_layer(x, mean_field_inference=mean_field_inference)

            # Compute KL
            self.kl = self.rnn_layer.kl + self.output_layer.kl

            return output
        elif "variational" in self.model:
            # apply variational dropout to input
            x = self.recurrent_dropout_layer(
                x, self.dropout, mean_field_inference=mean_field_inference
            )

            list_hidden = []
            for layer_idx, rnn_layer in enumerate(self.rnn_layers):
                x, hidden = rnn_layer(x, mean_field_inference=mean_field_inference)
                list_hidden.append(hidden)

                # Apply Variational dropout between recurrent layers
                if layer_idx != len(self.rnn_layers) - 1:
                    x = self.recurrent_dropout_layer(
                        x, self.dropout, mean_field_inference=mean_field_inference
                    )

            x = x.mean(0)
            # x is (batch, hidden_size)

            # apply dropout
            x = self.output_dropout_layer(
                x, self.dropout, mean_field_inference=mean_field_inference
            )
            # Final projection layer
            output = self.output_layer(x)

            return output

        else:
            x, hidden = self.rnn_layer(x)
            x = x.mean(0)
            # Final projection layer
            output = self.output_layer(x)

            return output


def evaluate_accuracy(X, Y, list_batches, net, device):
    numerator = 0.
    denominator = 0.

    net.eval()

    with torch.no_grad():

        for idxs in list_batches:

            s, e = idxs[0], idxs[-1] + 1

            X_batch, Y_batch = X[:, s:e, :], Y[s:e]

            X_batch = torch.from_numpy(X_batch).to(device)
            Y_batch = torch.from_numpy(Y_batch).to(device)

            output = net(X_batch, mean_field_inference=False)

            output = output.detach().cpu().numpy()
            predictions = np.argmax(output, axis=-1)
            numerator += np.sum(predictions == Y_batch.detach().cpu().numpy())
            denominator += X_batch.shape[1]

        return 100 * numerator / denominator


def evaluate_random(net, X_test, device):
    net.eval()

    with torch.no_grad():

        # Random data
        X_batch = torch.from_numpy(
            np.random.uniform(0, 1, (28, 1000, 28)).astype(np.float32)
        ).to(device)

        probs_MFE = (
            F.softmax(net(X_batch, mean_field_inference=True), dim=-1)
            .detach()
            .cpu()
            .numpy()
        )
        entropy_random_MFE = -np.mean((probs_MFE * np.log(probs_MFE)).sum(axis=1))

        arr_probs = []
        entropy = 0

        for k in range(200):

            probs = F.softmax(net(X_batch), dim=-1).detach().cpu().numpy()
            arr_probs.append(probs)

        probs = np.stack(arr_probs)
        entropy = probs.reshape((-1, 10))
        entropy_random_MC = -np.mean((entropy * np.log(entropy)).sum(axis=1))
        med_probs = np.median(np.stack(probs), axis=0)

        non_preds_random = len(np.where(np.max(med_probs, axis=-1) < 0.2)[0])

        # Real data
        batch_idxs = np.random.choice(np.arange(X_test.shape[1]), 1000, replace=False)
        X_batch = torch.from_numpy(np.ascontiguousarray(X_test[:, batch_idxs, :])).to(
            device
        )

        probs_MFE = (
            F.softmax(net(X_batch, mean_field_inference=True), dim=-1)
            .detach()
            .cpu()
            .numpy()
        )
        entropy_real_MFE = -np.mean((probs_MFE * np.log(probs_MFE)).sum(axis=1))

        arr_probs = []
        entropy = 0

        for k in range(200):

            probs = F.softmax(net(X_batch), dim=-1).detach().cpu().numpy()
            arr_probs.append(probs)

        probs = np.stack(arr_probs)
        entropy = probs.reshape((-1, 10))
        entropy_real_MC = -np.mean((entropy * np.log(entropy)).sum(axis=1))

        med_probs = np.median(np.stack(probs), axis=0)
        non_preds_real = len(np.where(np.max(med_probs, axis=-1) < 0.2)[0])

    return (
        non_preds_random,
        non_preds_real,
        entropy_random_MC - entropy_real_MC,
        entropy_random_MFE - entropy_real_MFE,
    )


def plot_preds(net, X_test, Y_test, device, epoch=None):

    x_one = X_test[:, Y_test == 1, :][:, 0, :]
    x_one = torch.from_numpy(x_one.reshape((1, 28, 28))).to("cpu")

    toPIL = torchvision.transforms.ToPILImage()

    def rotate(x, r):
        return np.array(torchvision.transforms.functional.rotate(toPIL(x), r))

    list_r = [0, 10, 20, 50]
    list_img = [rotate(x_one, r) for r in list_r]

    plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(4, 4, hspace=0.2)
    for i in range(len(list_img)):

        # Image
        ax = plt.subplot(gs[2 * i])
        ax.imshow(list_img[i], cmap="Greys_r")
        ax.set_xticks([])
        ax.set_yticks([])

        # Histo
        ax = plt.subplot(gs[2 * i + 1])
        x = torch.from_numpy(list_img[i]).to(device).view(28, 1, 28).float() / 256
        arr_probs = []
        for k in range(200):

            probs = F.softmax(net(x.to(device)), dim=-1).detach().cpu().numpy()
            arr_probs.append(probs)
        arr_probs = np.concatenate(arr_probs, 0)

        values, bin_edges = np.histogram(np.ravel(arr_probs), bins=25)

        for k in range(arr_probs.shape[-1]):
            ax.hist(
                arr_probs[:, k],
                color=f"C{k}",
                bins=bin_edges,
                histtype="step",
                label=f"Prob {k}",
            )
        ax.set_xlim([-0.1, 1.1])
        ax.set_yscale("log")
        ymin, ymax = ax.get_ylim()
        ax.set_ylim([ymin, 1.5 * np.max(values)])
        if i == len(list_img) - 1:
            ax.legend(bbox_to_anchor=(1.4, 1), fontsize=16)

    # Random images
    list_img = [np.random.uniform(0, 1, (28, 28)).astype(np.float32) for r in range(4)]

    for i in range(len(list_img), 2 * len(list_img)):

        # Image
        ax = plt.subplot(gs[2 * i])
        ax.imshow(list_img[i - len(list_img)], cmap="Greys_r")
        ax.set_xticks([])
        ax.set_yticks([])

        # Histo
        ax = plt.subplot(gs[2 * i + 1])
        x = (
            torch.from_numpy(list_img[i - len(list_img)])
            .to(device)
            .view(28, 1, 28)
            .float()
            / 256
        )
        arr_probs = []
        for k in range(200):

            probs = F.softmax(net(x.to(device)), dim=-1).detach().cpu().numpy()
            arr_probs.append(probs)
        arr_probs = np.concatenate(arr_probs, 0)

        values, bin_edges = np.histogram(np.ravel(arr_probs), bins=25)

        for k in range(arr_probs.shape[-1]):
            ax.hist(
                arr_probs[:, k],
                color=f"C{k}",
                bins=bin_edges,
                histtype="step",
                label=f"Prob {k}",
            )
        ax.set_xlim([-0.1, 1.1])
        ax.set_yscale("log")
        ymin, ymax = ax.get_ylim()
        ax.set_ylim([ymin, 1.5 * np.max(values)])
        if i >= 2 * len(list_img) - 2:
            ax.set_xlabel("Predicted Probability", fontsize=16)

    plt.subplots_adjust(
        left=0, right=0.9, bottom=0.1, top=0.98, wspace=0.0, hspace=0.02
    )
    title = (
        f"figmnist/RNN_{args.model}/fig.png"
        if epoch is None
        else f"figmnist/RNN_{args.model}/RNN_{args.model}_epoch_{epoch}.png"
    )
    plt.savefig(title)
    plt.clf()
    plt.close()


def train(args):

    shutil.rmtree(f"figmnist/RNN_{args.model}", ignore_errors=True)
    Path(f"figmnist/RNN_{args.model}").mkdir(exist_ok=True, parents=True)

    input_size, hidden_size, num_layers, bidirectional = 28, 256, 1, False
    prior = Prior(args.pi, args.log_sigma1, args.log_sigma2)
    mu_lower = -args.mu
    mu_upper = args.mu
    rho_lower = math.log(math.exp(prior.sigma_mix / args.scale_lower) - 1.0)
    rho_upper = math.log(math.exp(prior.sigma_mix / args.scale_upper) - 1.0)

    net = RNN(
        args.model,
        args.dropout,
        input_size,
        hidden_size,
        num_layers,
        bidirectional,
        prior,
        mu_lower,
        mu_upper,
        rho_lower,
        rho_upper,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        net.cuda()

    data = format_mnist()

    X_train = np.ascontiguousarray(
        data["X_train"][:].reshape((-1, 28, 28)).transpose(1, 0, 2)
    )
    X_test = np.ascontiguousarray(
        data["X_test"][:].reshape((-1, 28, 28)).transpose(1, 0, 2)
    )

    Y_train = data["Y_train"][:]
    Y_test = data["Y_test"][:]

    num_elem = X_train.shape[1]
    batch_size = 128
    num_train_batches = num_elem / batch_size
    list_train_batches = np.array_split(np.arange(num_elem), num_train_batches)

    num_elem = X_test.shape[1]
    batch_size = 128
    num_test_batches = num_elem / batch_size
    list_test_batches = np.array_split(np.arange(num_elem), num_test_batches)

    weight_decay = args.weight_decay if "variational" in args.model else 0
    optimizer = torch.optim.Adam(net.parameters(), lr=1E-3, weight_decay=weight_decay)

    train_acc = []
    test_acc = []

    criterion = nn.CrossEntropyLoss(reduction="sum")

    desc = ""

    plot_preds(net, X_test, Y_test, device, epoch=0)

    for epoch in range(10):

        list_loss = []
        list_loss_kl = []

        net.train()

        for train_idxs in tqdm(list_train_batches, desc=desc):

            optimizer.zero_grad()

            s, e = train_idxs[0], train_idxs[-1] + 1

            X_batch, Y_batch = X_train[:, s:e, :], Y_train[s:e]

            X_batch = torch.from_numpy(X_batch).to(device)
            Y_batch = torch.from_numpy(Y_batch).to(device)

            # Forward pass
            output = net(X_batch)

            # Get the loss
            likelihood_loss = criterion(output, Y_batch) / X_batch.shape[1]

            if "bayes" in args.model:
                kl = net.kl / (num_train_batches * X_batch.shape[1])
            else:
                kl = torch.zeros_like(likelihood_loss)

            loss = kl + likelihood_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            list_loss.append(likelihood_loss.item())
            list_loss_kl.append(kl.item())

        test_accuracy = evaluate_accuracy(
            X_train, Y_train, list_train_batches, net, device
        )
        train_accuracy = evaluate_accuracy(
            X_test, Y_test, list_test_batches, net, device
        )
        random_non_pred, test_non_pred, delta_entropy_MC, delta_entropy_MFE = evaluate_random(
            net, X_test, device
        )
        train_acc.append(np.asscalar(train_accuracy))
        test_acc.append(np.asscalar(test_accuracy))
        desc = (
            "Epoch %s. Loss: %.2g, KL: %.2g, Train_acc %.2g, "
            "Test_acc %.2g, Random non_pred %s/1000, Test non_pred %s/1000 dEntropy MC %.2g dEntropy MFE %.2g"
            % (
                epoch + 1,
                np.mean(list_loss),
                np.mean(list_loss_kl),
                train_accuracy,
                test_accuracy,
                random_non_pred,
                test_non_pred,
                delta_entropy_MC,
                delta_entropy_MFE,
            )
        )

        plot_preds(net, X_test, Y_test, device, epoch=epoch + 1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="PyTorch PTB Bayesian RNN language model"
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["bayes", "variational", "standard"],
        default="bayes",
        help="type of NN model",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="dropout for variational"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-6, help="dropout for variational"
    )
    parser.add_argument("--pi", type=float, default=0.25, help="prior mixing")
    parser.add_argument(
        "--log_sigma1", type=float, default=-1., help="prior log sigma1"
    )
    parser.add_argument(
        "--log_sigma2", type=float, default=-7., help="prior log sigma2"
    )
    parser.add_argument("--mu", type=float, default=0.05, help="init for bayesian locs")
    parser.add_argument(
        "--scale_lower", type=float, default=4., help="prior scale init lower"
    )
    parser.add_argument(
        "--scale_upper", type=float, default=2., help="prior scale init upper"
    )
    parser.add_argument("--seed", type=int, default=1111, help="random seed")

    args = parser.parse_args()

    # Set the random seed manually for reproducibility.
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(args.seed)

    print()
    print()
    print(" ".join([f"{k}:{v}" for k, v in args.__dict__.items()]))
    train(args)
