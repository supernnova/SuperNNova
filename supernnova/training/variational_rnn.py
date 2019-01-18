import torch


class VariationalRNN(torch.nn.Module):

    def __init__(self, input_size, settings):

        super(VariationalRNN, self).__init__()

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
                dropout=0.,  # no dropout: we later create a specific layer for that
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

            # In vanilla standard we have: out, hidden = rnn(X)
            # hidden is built as follows:
            # for each layer, for each direction, take the last hidden state
            # concatenate the results. obtain hidden (num_layers * num_directions, B, D)
            # here, we need to do it manually as the recurrent layers are written one by one
            # We now carry out the ``concatenate`` operation.
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


class WeightDrop(torch.nn.Module):
    """
    WeightDrop from https://github.com/salesforce/awd-lstm-lm

    - Removed the variational input parameter as we will always use WeightDrop in variational mode

    """

    def __init__(self, module, weights, dropout):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self._setup()

    def dummy_function(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        return

    def _setup(self):
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.dummy_function

        for name_w in self.weights:
            print(f"Applying weight drop of {self.dropout} to {name_w}")
            w = getattr(self.module, name_w)  # take this param of the module
            del self.module._parameters[name_w]
            self.module.register_parameter(
                name_w + "_raw", torch.nn.Parameter(w.data)
            )  # recreate same data with different name

    def _setweights(self, mean_field_inference=False):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + "_raw")
            w = None
            mask = torch.ones(raw_w.size(0), 1)
            device = raw_w.device
            mask = mask.to(device)
            # forcing it to be applied training and validation
            if mean_field_inference is False:
                mask = torch.nn.functional.dropout(mask, p=self.dropout, training=True)
            mask = mask.expand_as(raw_w)
            # applying same mask to all elements of the batch
            # h [batch, xdim]
            w = mask * raw_w
            setattr(self.module, name_w, w)

    def forward(self, *args, mean_field_inference=False):
        # Apply dropout to weights of module
        self._setweights(mean_field_inference=mean_field_inference)

        return self.module.forward(*args)


def embedded_dropout(embed, words, dropout, mean_field_inference=False):
    """
    embedded_dropout from https://github.com/salesforce/awd-lstm-lm
    with some modifications like the mean field inference flag

    """
    if mean_field_inference:
        masked_embed_weight = embed.weight
    else:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(
            1 - dropout
        ).expand_as(embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1

    X = torch.nn.functional.embedding(
        words,
        masked_embed_weight,
        padding_idx,
        embed.max_norm,
        embed.norm_type,
        embed.scale_grad_by_freq,
        embed.sparse,
    )

    return X
