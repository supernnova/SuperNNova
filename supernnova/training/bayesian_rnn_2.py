import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


def log_norm(x, min_clip, mean, std, F=torch):
    """
    """

    x = (F.log(x - min_clip + 1e-5) - mean) / std
    return x


def inverse_log_norm(x, min_clip, mean, std, F=torch):

    x = F.exp(x * std + mean) + min_clip - 1e-5

    return x


class BayesianRNN(torch.nn.Module):
    def __init__(self, input_size, settings):
        super().__init__()

        # Params
        self.layer_type = settings.layer_type
        self.hidden_size = settings.hidden_dim
        self.output_size = settings.nb_classes
        self.num_layers = settings.num_layers
        self.bidirectional = settings.bidirectional
        self.use_cuda = settings.use_cuda
        self.rnn_output_option = settings.rnn_output_option

        self.prior = Prior(settings.pi, settings.log_sigma1, settings.log_sigma2)
        self.prior_output = Prior(
            settings.pi, settings.log_sigma1_output, settings.log_sigma2_output
        )

        bidirectional_factor = 2 if self.bidirectional is True else 1
        last_input_size = (
            self.hidden_size * bidirectional_factor
            if self.rnn_output_option == "mean"
            else self.hidden_size * bidirectional_factor * self.num_layers
        )

        init_recurrent = {
            "mu_lower": -0.05,
            "mu_upper": 0.05,
            "rho_lower": math.log(
                math.exp(self.prior.sigma_mix / settings.rho_scale_lower) - 1.0
            ),
            "rho_upper": math.log(
                math.exp(self.prior.sigma_mix / settings.rho_scale_upper) - 1.0
            ),
        }

        init = {
            "mu_lower": -0.05,
            "mu_upper": 0.05,
            "rho_lower": math.log(
                math.exp(self.prior_output.sigma_mix / settings.rho_scale_lower_output)
                - 1.0
            ),
            "rho_upper": math.log(
                math.exp(self.prior_output.sigma_mix / settings.rho_scale_upper_output)
                - 1.0
            ),
        }

        # Define layers
        self.rnn_layer = BayesLSTM(
            input_size,
            self.hidden_size,
            num_layers=self.num_layers,
            dropout=0,
            bidirectional=self.bidirectional,
            batch_first=False,
            bias=True,
            prior=self.prior,
            **init_recurrent,
        )
        self.output_layer = BayesLinear(
            last_input_size, self.output_size, self.prior_output, **init
        )

        self.kl = None

    def forward(self, x, mean_field_inference=False):
        # out = packed output from last layer
        # out has dim (seq_len, batch_size, hidden_size) when unpacked
        # hidden = (hn, cn) for lstm
        # hidden = hn for GRU and RNN
        # hn has dim (num_layers * num_directions, batch, hidden_size)
        # cn has dim (num_layers * num_directions, batch, hidden_size)
        # assuming num_directions = 1, num_layers = 2 :
        # hn[-1, -1] == out[len, -1] where len is the len of the seq at batch index == -1

        x, hidden = self.rnn_layer(x, mean_field_inference=mean_field_inference)

        # Output options
        # Standard: all layers, only end of pass
        #    - take last pass in all layers (hidden)
        #    - reshape and apply dropout
        #    - use h20 to obtain output (h2o input: hidden_size*num_layers*bi)
        # Mean: last layer, mean on sequence
        #    - take packed output from last layer (out) that contains all time steps for the last layer
        #    - find where padding was done and create a mask for those values, apply this mask
        #    - take a mean for the whole sequence (time_steps)
        #    - use h2o to obtain output (beware! it is only one layer deep since it is the last one only)

        if self.rnn_output_option == "standard":
            # Special case for lstm where hidden = (h, c)
            if self.layer_type == "lstm":
                hn = hidden[0]
            else:
                hn = hidden

            hn = hn.permute(1, 2, 0).contiguous()
            # hn is (num_layers * num_directions, batch, hidden_size)
            batch_size = hn.shape[0]
            # hn now is (batch, hidden size, num_layers * num_directions)
            x = hn.view(batch_size, -1)
            # x is (batch, hidden size * num_layers * num_directions)

        if self.rnn_output_option == "mean":
            if isinstance(x, torch.nn.utils.rnn.PackedSequence):
                x, lens = torch.nn.utils.rnn.pad_packed_sequence(x)
                # x is (seq_len, batch, hidden size)

                # take mean over seq_len
                x = x.sum(0) / lens.unsqueeze(-1).float().to(x.device)
                # x is (batch, hidden_size)
            else:
                x = x.mean(0)
                # x is (batch, hidden_size)

        # Final projection layer
        output = self.output_layer(x, mean_field_inference=mean_field_inference)

        # Compute KL
        self.kl = self.rnn_layer.kl + self.output_layer.kl

        return output


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

    def _setweights(self, mean_field_inference=False):
        kl = 0
        for name, _ in self.module.named_parameters():
            mu = getattr(self, f"{name}_mu")
            rho = getattr(self, f"{name}_rho")

            sigma = F.softplus(rho) + 1e-5

            if mean_field_inference is False:
                eps = rho.data.new(rho.size()).normal_(0.0, 1.0)
                w = mu + sigma * eps
                kl += compute_KL(w, mu, sigma, self.prior)
            else:
                w = mu

            self.module._parameters[name] = w

        self.kl = kl

    def forward(self, *args, mean_field_inference=False):
        self._setweights()
        self.module.flatten_parameters()
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

    def forward(self, x, mean_field_inference=False):

        self.kl = 0

        mu = self.mu

        if mean_field_inference is False:
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

    if x.is_cuda:
        n1 = torch.distributions.Normal(
            torch.tensor([0.0]).cuda(), torch.tensor([prior.sigma1]).cuda()
        )
        n2 = torch.distributions.Normal(
            torch.tensor([0.0]).cuda(), torch.tensor([prior.sigma2]).cuda()
        )
    else:
        n1 = torch.distributions.Normal(0.0, prior.sigma1)
        n2 = torch.distributions.Normal(0.0, prior.sigma2)

    mix1 = torch.sum(n1.log_prob(x)) + math.log(prior.pi_mixture)
    mix2 = torch.sum(n2.log_prob(x)) + math.log(1.0 - prior.pi_mixture)
    prior_mix = torch.stack([mix1, mix2])
    log_prior = torch.logsumexp(prior_mix, 0)

    return log_posterior - log_prior
