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

from supernnova.training.bayesian_rnn import BayesLinear, BayesBiasLinear, Prior
from supernnova.training.variational_rnn import VariationalDropout


class NN(nn.Module):

    def __init__(self, model, dropout, prior, mu_lower, mu_upper, rho_lower, rho_upper):
        super(NN, self).__init__()

        self.model = model
        self.dropout = dropout

        if model == "bayes_bias":
            self.fc1 = BayesBiasLinear(
                784, 400, prior, mu_lower, mu_upper, rho_lower, rho_upper
            )
            self.fc2 = BayesLinear(
                400, 10, prior, mu_lower, mu_upper, rho_lower, rho_upper
            )
        elif model == "bayes":
            self.fc1 = BayesBiasLinear(
                784, 400, prior, mu_lower, mu_upper, rho_lower, rho_upper
            )
            self.fc2 = BayesBiasLinear(
                400, 10, prior, mu_lower, mu_upper, rho_lower, rho_upper
            )
        elif model == "variational":
            self.dropout_layer = VariationalDropout()
            self.fc1 = nn.Linear(784, 400)
            self.fc2 = nn.Linear(400, 10)

        else:
            self.fc1 = nn.Linear(784, 400)
            self.fc2 = nn.Linear(400, 10)

    def forward(self, x, mean_field_inference=False):
        if "bayes" in self.model:
            output = self.fc1(x, mean_field_inference=mean_field_inference)
            output = F.relu(output)
            output = self.fc2(output, mean_field_inference=mean_field_inference)
            return output

        elif self.model == "variational":
            x = self.dropout_layer(
                x, self.dropout, mean_field_inference=mean_field_inference
            )
            output = self.fc1(x)
            output = F.relu(output)
            x = self.dropout_layer(
                x, self.dropout, mean_field_inference=mean_field_inference
            )
            output = self.fc2(output)
            return output

        else:
            output = self.fc1(x)
            output = F.relu(output)
            output = self.fc2(output)
            return output


def evaluate_accuracy(X, Y, list_batches, net, device):
    numerator = 0.
    denominator = 0.

    with torch.no_grad():

        for idxs in list_batches:

            s, e = idxs[0], idxs[-1] + 1

            X_batch, Y_batch = X[s:e], Y[s:e]

            X_batch = torch.from_numpy(X_batch).to(device)
            Y_batch = torch.from_numpy(Y_batch).to(device)

            output = net(X_batch, mean_field_inference=True).detach().cpu().numpy()
            predictions = np.argmax(output, axis=-1)
            numerator += np.sum(predictions == Y_batch.detach().cpu().numpy())
            denominator += X_batch.shape[0]

        return 100 * numerator / denominator


def evaluate_random(net, X_test, device):

    with torch.no_grad():

        # Random data
        X_batch = torch.from_numpy(
            np.random.uniform(0, 1, (1000, 784)).astype(np.float32)
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
        batch_idxs = np.random.choice(np.arange(X_test.shape[0]), 1000, replace=False)
        X_batch = torch.from_numpy(np.ascontiguousarray(X_test[batch_idxs])).to(device)

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

    x_one = X_test[Y_test == 1][0]
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
        x = torch.from_numpy(list_img[i]).to(device).view(1, -1).float() / 256
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
            torch.from_numpy(list_img[i - len(list_img)]).to(device).view(1, -1).float()
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
        f"figmnist/MLP_{args.model}/fig.png"
        if epoch is None
        else f"figmnist/MLP_{args.model}/MLP_{args.model}_epoch_{epoch}.png"
    )
    plt.savefig(title)
    plt.clf()
    plt.close()


def train(args):

    shutil.rmtree(f"figmnist/MLP_{args.model}", ignore_errors=True)
    Path(f"figmnist/MLP_{args.model}").mkdir(exist_ok=True, parents=True)

    prior = Prior(args.pi, args.log_sigma1, args.log_sigma2)
    mu_lower = -args.mu
    mu_upper = args.mu
    rho_lower = math.log(math.exp(prior.sigma_mix / args.scale_lower) - 1.0)
    rho_upper = math.log(math.exp(prior.sigma_mix / args.scale_upper) - 1.0)

    net = NN(args.model, args.dropout, prior, mu_lower, mu_upper, rho_lower, rho_upper)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        net.cuda()

    criterion = nn.CrossEntropyLoss(reduction="sum")

    data = format_mnist()

    X_train = data["X_train"][:]
    X_test = data["X_test"][:]

    Y_train = data["Y_train"][:]
    Y_test = data["Y_test"][:]

    num_elem = X_train.shape[0]
    batch_size = 128
    num_train_batches = num_elem / batch_size
    list_train_batches = np.array_split(np.arange(num_elem), num_train_batches)

    num_elem = X_test.shape[0]
    batch_size = 128
    num_test_batches = num_elem / batch_size
    list_test_batches = np.array_split(np.arange(num_elem), num_test_batches)

    weight_decay = args.weight_decay if "variational" in args.model else 0
    optimizer = torch.optim.Adam(net.parameters(), lr=1E-3, weight_decay=weight_decay)

    train_acc = []
    test_acc = []

    desc = ""

    plot_preds(net, X_test, Y_test, device, epoch=0)

    for epoch in range(10):

        list_loss = []
        list_loss_kl = []

        for train_idxs in tqdm(list_train_batches, desc=desc):

            optimizer.zero_grad()

            s, e = train_idxs[0], train_idxs[-1] + 1

            X_batch, Y_batch = X_train[s:e], Y_train[s:e]

            X_batch = torch.from_numpy(X_batch).to(device)
            Y_batch = torch.from_numpy(Y_batch).to(device)

            # Forward pass
            output = net(X_batch)

            # Get the loss
            likelihood_loss = criterion(output, Y_batch) / (X_batch.shape[0])
            if "bayes" in args.model:
                kl = (net.fc1.kl + net.fc2.kl) / (num_train_batches * X_batch.shape[0])
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
        choices=["bayes", "bayes_bias", "variational", "standard"],
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
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print()
    print()
    print(" ".join([f"{k}:{v}" for k, v in args.__dict__.items()]))
    train(args)
