import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(torch.nn.Module):
    def __init__(self, input_size, settings):

        super().__init__()

        # Params
        self.input_size = input_size
        self.layer_type = settings.layer_type
        self.output_size = settings.nb_classes
        self.hidden_size = settings.hidden_dim
        self.num_layers = settings.num_layers
        self.bidirectional = settings.bidirectional
        self.dropout = settings.dropout
        self.use_cuda = settings.use_cuda
        self.rnn_output_option = settings.rnn_output_option

        bidirectional_factor = 2 if self.bidirectional is True else 1
        last_input_size = (
            self.hidden_size * bidirectional_factor
            if self.rnn_output_option == "mean"
            else self.hidden_size * bidirectional_factor * self.num_layers
        )

        # Need to create recurrent layers one by one because we use variational dropout
        self.rnn_layers = []
        for i in range(self.num_layers):
            if i == 0:
                input_size = self.input_size
            else:
                previous_layer = getattr(self, f"rnn_layer{i - 1}")
                input_size = previous_layer.module.hidden_size * bidirectional_factor

            # Create recurrent layer
            layer = getattr(torch.nn, self.layer_type.upper())(
                input_size,
                self.hidden_size,
                num_layers=1,
                dropout=0.0,  # no dropout: we later create a specific layer for that
                bidirectional=self.bidirectional,
            )
            # Apply weight drop
            layer = WeightDrop(layer, ["weight_hh_l0"], dropout=self.dropout)
            # Set layer as attribute
            setattr(self, f"rnn_layer{i}", layer)
            self.rnn_layers.append(layer)

        self.recurrent_dropout_layer = VariationalRecurrentDropout()
        self.output_dropout_layer = VariationalDropout()
        self.output_layer = torch.nn.Linear(last_input_size, self.output_size)

    def apply_recurrent_variational_dropout(
        self, x, dropout_value, mean_field_inference=False
    ):

        # apply dropout to input
        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            # Unpack
            x, lengths = torch.nn.utils.rnn.pad_packed_sequence(x)
            # Apply locked dropout
            x = self.recurrent_dropout_layer(
                x, dropout_value, mean_field_inference=mean_field_inference
            )
            # Repack
            x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths)
        else:
            x = self.recurrent_dropout_layer(
                x, dropout_value, mean_field_inference=mean_field_inference
            )

        return x

    def forward(self, x, mean_field_inference=False):

        # apply variational dropout to input
        x = self.apply_recurrent_variational_dropout(
            x, self.dropout, mean_field_inference
        )

        list_hidden = []
        for layer_idx, rnn_layer in enumerate(self.rnn_layers):
            x, hidden = rnn_layer(x, mean_field_inference=mean_field_inference)
            list_hidden.append(hidden)

            # Apply Variational dropout between recurrent layers
            if layer_idx != len(self.rnn_layers) - 1:
                x = self.apply_recurrent_variational_dropout(
                    x, self.dropout, mean_field_inference=mean_field_inference
                )

        if self.rnn_output_option == "standard":
            # Special case for lstm where hidden = (h, c)

            if self.layer_type == "lstm":
                hn = torch.cat([h[0] for h in list_hidden], dim=0)
            else:
                hn = torch.cat(list_hidden, dim=0)

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

        # apply dropout
        x = self.output_dropout_layer(
            x, self.dropout, mean_field_inference=mean_field_inference
        )
        # Final projection layer
        output = self.output_layer(x)

        return output


class VariationalRecurrentDropout(torch.nn.Module):
    """
    This is a renamed Locked Dropout from https://github.com/salesforce/awd-lstm-lm

    - We added a mean_field_inference flag to carry out mean field inference

    We do this so that choosing whether to use dropout or not at inference time
    is explicit

    """

    def __init__(self):
        super().__init__()

    def forward(self, x, dropout, mean_field_inference=False):
        if mean_field_inference is True:
            return x
        else:
            m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
            mask = m / (1 - dropout)  # rescaling to account for dropout
            mask = mask.expand_as(x)

            return mask * x


class VariationalDropout(torch.nn.Module):
    """Re-implementation of torch.nn.modules.Dropout

    - training flag is always set to True
    - We added a mean_field_inference flag to carry out mean field inference

    We do this rather than using the training flag so that we can more explicitly decide
    to use dropout or not at inference time
    """

    def forward(self, x, dropout, mean_field_inference=False):
        training = True
        inplace = False
        if mean_field_inference is True:
            return x
        else:
            return torch.nn.functional.dropout(x, dropout, training, inplace)


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
            mask = dropout_mask(self.emb.weight.data, size, self.embed_p)
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
        if not self.training or self.p == 0.0:
            return x

        mask = (
            dropout_mask(x.data, (x.shape[0], 1, x.shape[2]), self.dropout)
            if self.batch_first
            else dropout_mask(x.data, (1, x.shape[1], x.shape[2]), self.dropout)
        )
        return x * mask

