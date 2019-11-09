import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


# #####################################
# Monte Carlo Dropout Bayesian Layers
# #####################################


def dropout_mask(x, size, dropout):
    return x.new(*size).bernoulli_(1 - dropout).div_(1 - dropout)


class EmbeddingDropout(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, dropout):
        super().__init__()

        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.dropout = dropout

    def forward(self, x):
        if self.training:
            size = (self.emb.weight.shape[0], 1)
            mask = dropout_mask(self.emb.weight.data, size, self.dropout)
            masked_embed = self.emb.weight * mask
        else:
            masked_embed = self.emb.weight
        return F.embedding(
            x,
            masked_embed,
            self.emb.padding_idx,
            self.emb.max_norm,
            self.emb.norm_type,
            self.emb.scale_grad_by_freq,
            self.emb.sparse,
        )


class WeightDropout(nn.Module):
    def __init__(self, module, dropout, layer_names):
        super().__init__()
        self.module = module
        self.dropout = dropout
        self.layer_names = layer_names

        for layer in self.layer_names:
            # Makes a copy of the weights of the selected layers.
            w = getattr(self.module, layer)
            self.register_parameter(f"{layer}_raw", nn.Parameter(w.data))
            self.module._parameters[layer] = F.dropout(
                w, p=self.dropout, training=False
            )

    def _setweights(self):
        "Apply dropout to the raw weights."
        for layer in self.layer_names:
            raw_w = getattr(self, f"{layer}_raw")
            self.module._parameters[layer] = F.dropout(
                raw_w, p=self.dropout, training=self.training
            )

    def forward(self, *args):
        self._setweights()
        with warnings.catch_warnings():
            # To avoid the warning that comes because the weights aren't flattened.
            warnings.simplefilter("ignore")
            return self.module.forward(*args)

    def reset(self):
        for layer in self.layer_names:
            raw_w = getattr(self, f"{layer}_raw")
            self.module._parameters[layer] = F.dropout(
                raw_w, p=self.dropout, training=False
            )
        if hasattr(self.module, "reset"):
            self.module.reset()


class RNNDropout(nn.Module):
    def __init__(self, dropout, batch_first=False):
        super().__init__()
        self.dropout = dropout
        self.batch_first = batch_first

    def forward(self, x):
        if not self.training or self.dropout == 0.0:
            return x

        mask = (
            dropout_mask(x.data, (x.shape[0], 1, x.shape[2]), self.dropout)
            if self.batch_first
            else dropout_mask(x.data, (1, x.shape[1], x.shape[2]), self.dropout)
        )
        return x * mask


# #####################################
# Bayes By Backprop Bayesian layers
# #####################################


class BayesLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        batch_first=False,
        dropout=0.0,
        bidirectional=False,
        prior=None,
        mu_lower=-0.05,
        mu_upper=0.05,
        rho_lower=math.log(math.exp(1.0 / 4.0) - 1.0),
        rho_upper=math.log(math.exp(1.0 / 2.0) - 1.0),
    ):

        super().__init__()
        self.module = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
        )

        self.prior = prior
        self.kl = None
        for name, w in self.module.named_parameters():
            mu, rho = get_bbb_variable(
                w.shape, mu_lower, mu_upper, rho_lower, rho_upper
            )
            self.register_parameter(f"{name}_mu", mu)
            self.register_parameter(f"{name}_rho", rho)

            self.module._parameters[name] = mu

    def _setweights(self):
        kl = 0
        for name, _ in self.module.named_parameters():
            mu = getattr(self, f"{name}_mu")
            rho = getattr(self, f"{name}_rho")

            sigma = F.softplus(rho) + 1e-5

            if self.training:
                eps = rho.data.new(rho.size()).normal_(0.0, 1.0)
                w = mu + sigma * eps
                kl += compute_KL(w, mu, sigma, self.prior)
            else:
                w = mu

            self.module._parameters[name] = w

        self.kl = kl

    def forward(self, *args):
        self._setweights()
        with warnings.catch_warnings():
            # To avoid the warning that comes because the weights aren't flattened.
            warnings.simplefilter("ignore")
            return self.module.forward(*args)


class BayesLinear(nn.Module):
    def __init__(
        self, in_features, out_features, prior, mu_lower, mu_upper, rho_lower, rho_upper
    ):
        super(BayesLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior = prior

        mu, rho = get_bbb_variable(
            (out_features, in_features), mu_lower, mu_upper, rho_lower, rho_upper
        )

        bias = nn.Parameter(torch.Tensor(out_features))
        bias.data.fill_(0.0)

        self.mu = mu
        self.rho = rho
        self.bias = bias
        self.kl = None

    def forward(self, x):

        self.kl = 0

        mu = self.mu

        if self.training:
            sigma = F.softplus(self.rho) + 1e-5
            eps = mu.data.new(mu.size()).normal_(0.0, 1.0)
            weights = mu + eps * sigma
            # Compute KL divergence
            self.kl = compute_KL(weights, mu, sigma, self.prior)
        else:
            weights = mu

        logits = F.linear(x, weights, self.bias)

        return logits

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class BayesBiasLinear(nn.Module):
    def __init__(
        self, in_features, out_features, prior, mu_lower, mu_upper, rho_lower, rho_upper
    ):
        super(BayesBiasLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior = prior

        self.W_mu, self.W_rho = get_bbb_variable(
            (out_features, in_features), mu_lower, mu_upper, rho_lower, rho_upper
        )

        self.b_mu, self.b_rho = get_bbb_variable(
            (out_features,), mu_lower, mu_upper, rho_lower, rho_upper
        )

    def forward(self, x):

        self.kl = 0

        W_mu = self.W_mu
        b_mu = self.b_mu

        if self.training:

            W_sigma = F.softplus(self.W_rho) + 1e-5
            W_eps = W_mu.data.new(W_mu.size()).normal_(0.0, 1.0)
            weights = W_mu + W_eps * W_sigma

            b_sigma = F.softplus(self.b_rho) + 1e-5
            b_eps = b_mu.data.new(b_mu.size()).normal_(0.0, 1.0)
            biases = b_mu + b_eps * b_sigma

            # Compute KL divergence
            self.kl += compute_KL(weights, W_mu, W_sigma, self.prior)
            self.kl += compute_KL(biases, b_mu, b_sigma, self.prior)
        else:
            weights = W_mu
            biases = b_mu

        logits = F.linear(x, weights, biases)

        return logits


class BayesEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        prior,
        mu_lower,
        mu_upper,
        rho_lower,
        rho_upper,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.prior = prior
        self.max_norm = None
        self.norm_type = 2
        self.scale_grad_by_freq = False
        self.sparse = False
        self.padding_idx = -1

        mu, rho = get_bbb_variable(
            [num_embeddings, embedding_dim], mu_lower, mu_upper, rho_lower, rho_upper
        )

        self.mu = mu
        self.rho = rho
        self.kl = None

    def forward(self, x):

        self.kl = 0

        mu = self.mu

        if self.training:
            sigma = F.softplus(self.rho) + 1e-5
            eps = mu.data.new(mu.size()).normal_(0.0, 1.0)
            weights = mu + eps * sigma
            # Compute KL divergence
            self.kl = compute_KL(weights, mu, sigma, self.prior)
        else:
            weights = mu

        after_embed = F.embedding(
            x,
            weights,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

        return after_embed

    def __repr__(self):
        s = "{name}({num_embeddings}, {embedding_dim}"
        if self.padding_idx is not None:
            s += ", padding_idx={padding_idx}"
        if self.max_norm is not None:
            s += ", max_norm={max_norm}"
        if self.norm_type != 2:
            s += ", norm_type={norm_type}"
        if self.scale_grad_by_freq is not False:
            s += ", scale_grad_by_freq={scale_grad_by_freq}"
        if self.sparse is not False:
            s += ", sparse=True"
        s += ")"
        return s.format(name=self.__class__.__name__, **self.__dict__)


class Prior(object):
    def __init__(self, pi=0.25, log_sigma1=-1.0, log_sigma2=-7.0):
        self.pi_mixture = pi
        self.log_sigma1 = log_sigma1
        self.log_sigma2 = log_sigma2
        self.sigma1 = math.exp(log_sigma1)
        self.sigma2 = math.exp(log_sigma2)

        self.sigma_mix = math.sqrt(
            pi * math.pow(self.sigma1, 2) + (1.0 - pi) * math.pow(self.sigma2, 2)
        )


def get_bbb_variable(shape, mu_lower, mu_upper, rho_lower, rho_upper):

    mu = nn.Parameter(torch.Tensor(*shape))
    rho = nn.Parameter(torch.Tensor(*shape))

    # Initialize
    mu.data.uniform_(mu_lower, mu_upper)
    rho.data.uniform_(rho_lower, rho_upper)

    return mu, rho


def compute_KL(x, mu, sigma, prior):

    posterior = torch.distributions.Normal(mu.view(-1), sigma.view(-1))
    log_posterior = posterior.log_prob(x.view(-1)).sum()

    n1 = torch.distributions.Normal(0.0, prior.sigma1)
    n2 = torch.distributions.Normal(0.0, prior.sigma2)

    mix1 = torch.sum(n1.log_prob(x)) + math.log(prior.pi_mixture)
    mix2 = torch.sum(n2.log_prob(x)) + math.log(1.0 - prior.pi_mixture)
    prior_mix = torch.stack([mix1, mix2])
    log_prior = torch.logsumexp(prior_mix, 0)

    return log_posterior - log_prior
