import math
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from supernnova.modules.bayesian_layers import (
    BayesBiasLinear,
    BayesEmbedding,
    BayesLSTM,
    Prior,
)


def log_norm(x, min_clip, mean, std, F=torch):
    """
    """

    x = (F.log(x - min_clip + 1e-5) - mean) / std
    return x


def inverse_log_norm(x, min_clip, mean, std, F=torch):

    x = F.exp(x * std + mean) + min_clip - 1e-5

    return x


class Model(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers,
        dropout,
        num_embeddings,
        embedding_dim,
        normalize=False,
        pi=0.75,
        log_sigma1=-1.0,
        log_sigma2=-7.0,
    ):
        super().__init__()

        self.normalize = normalize

        self.prior = Prior(pi, log_sigma1, log_sigma2)

        init_recurrent = {
            "mu_lower": -0.05,
            "mu_upper": 0.05,
            "rho_lower": math.log(math.exp(self.prior.sigma_mix / 4.0) - 1.0),
            "rho_upper": math.log(math.exp(self.prior.sigma_mix / 2.0) - 1.0),
        }

        init_non_recurrent = {
            "mu_lower": -0.05,
            "mu_upper": 0.05,
            "rho_lower": math.log(math.exp(self.prior.sigma_mix / 2.0) - 1.0),
            "rho_upper": math.log(math.exp(self.prior.sigma_mix / 1.0) - 1.0),
        }

        self.embedding = BayesEmbedding(
            num_embeddings, embedding_dim, self.prior, **init_non_recurrent
        )

        # Define layers
        self.rnn = BayesLSTM(
            input_size + embedding_dim,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
            bias=True,
            prior=self.prior,
            **init_recurrent
        )
        self.output_layer = BayesBiasLinear(
            hidden_size * 2, output_size, self.prior, **init_non_recurrent
        )

        if self.normalize:
            self.register_buffer("flux_norm", torch.zeros(3))
            self.register_buffer("fluxerr_norm", torch.zeros(3))
            self.register_buffer("delta_time_norm", torch.zeros(3))

        self.kl = None

    def forward(self, x_flux, x_fluxerr, x_flt, x_time, x_mask, x_meta=None):

        # TODO constants ?
        x_flux = x_flux.clamp(-100)
        x_fluxerr = x_fluxerr.clamp(-100)

        if self.normalize:
            x_flux = log_norm(x_flux, *self.flux_norm)
            x_fluxerr = log_norm(x_fluxerr, *self.flux_norm)
            x_time = log_norm(x_time, *self.flux_norm)

        x_flt = self.embedding(x_flt)

        x = torch.cat([x_flux, x_fluxerr, x_time, x_flt], dim=-1)
        # x is (B, L, D)
        # mask is (B, L)
        B, L, _ = x.shape
        if x_meta is not None:
            x_meta = x_meta.unsqueeze(1).expand(B, L, -1)
            x = torch.cat([x, x_meta], dim=-1)

        lengths = x_mask.sum(dim=-1).long()
        x_packed = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        # Pass it to the RNN
        hidden_packed, (x, _) = self.rnn(x_packed)
        # undo PackedSequence
        hidden, _ = pad_packed_sequence(hidden_packed, batch_first=True)
        # hidden is (B, L, D)

        # Take masked mean
        x = (hidden * x_mask.unsqueeze(-1).float()).sum(1) / x_mask.float().unsqueeze(
            -1
        ).sum(1)

        # Final projection layer
        output = self.output_layer(x)

        # Compute KL
        self.kl = self.rnn.kl + self.output_layer.kl + self.embedding.kl

        return output
