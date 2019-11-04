import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from supernnova.utils.training_utils import log_norm


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
    ):
        super(Model, self).__init__()

        self.normalize = normalize

        self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim)

        # Define layers
        self.rnn = torch.nn.LSTM(
            input_size + embedding_dim,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
            bias=True,
        )
        self.output_dropout_layer = torch.nn.Dropout(dropout)
        self.output_layer = torch.nn.Linear(hidden_size * 2, output_size)

        if self.normalize:
            self.register_buffer("flux_norm", torch.zeros(3))
            self.register_buffer("fluxerr_norm", torch.zeros(3))
            self.register_buffer("delta_time_norm", torch.zeros(3))

    def forward(self, x_flux, x_fluxerr, x_flt, x_time, x_mask, x_meta=None):

        if self.normalize:
            x_flux = log_norm(x_flux, *self.flux_norm)
            x_fluxerr = log_norm(x_fluxerr, *self.flux_norm)
            x_time = log_norm(x_time, *self.flux_norm)

        x_flt = self.embedding(x_flt)

        import ipdb

        ipdb.set_trace()

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
        hidden_packed, _ = self.rnn(x_packed)
        # undo PackedSequence
        hidden, _ = pad_packed_sequence(hidden_packed, batch_first=True)
        # hidden is (B, L, D)

        # Take masked mean
        x = (hidden * x_mask.unsqueeze(-1).float()).sum(1) / x_mask.unsqueeze(-1).sum(1)

        # apply dropout
        x = self.output_dropout_layer(x)
        # Final projection layer
        output = self.output_layer(x)

        return output
