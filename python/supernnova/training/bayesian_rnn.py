# class BayesRNNBase and BayesLSTM are developed based on RNNBase and LSTM from pytorch package (v2.0.1) (https://github.com/pytorch/pytorch/tree/v2.0.1)
# Source: https://github.com/pytorch/pytorch/blob/v2.0.1/torch/nn/modules/rnn.py

import math
import numbers
import warnings
from typing import Tuple, Optional

import torch
from torch import Tensor, _VF
from torch.nn import Parameter, Module
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence

import torch.nn.functional as F



# mean_field_inference: instead of sampling a weight as in W = N(mu, sigma)
# we always set W = mu

class BayesianRNN(Module):

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

        # Layers / nn objects
        if self.layer_type == "lstm":
            print("starting rnn layer ...")
            self.rnn_layer = BayesLSTM(
                input_size,
                self.hidden_size,
                self.prior,
                num_layers=self.num_layers,
                bidirectional=self.bidirectional,
                mu_lower=-0.05,
                mu_upper=0.05,
                rho_lower=math.log(
                    math.exp(self.prior.sigma_mix / settings.rho_scale_lower) - 1.0
                ),
                rho_upper=math.log(
                    math.exp(self.prior.sigma_mix / settings.rho_scale_upper) - 1.0
                ),
            )
            print("type of rnn layer: ", type(self.rnn_layer))
            print("finish rnn layer...")
        else:
            raise ValueError("Unregistered BayesRNN mode: {}".format(self.layer_type.upper()))

        self.output_layer = BayesLinear(
            last_input_size,
            self.output_size,
            self.prior_output,
            mu_lower=-0.05,
            mu_upper=0.05,
            rho_lower=math.log(
                math.exp(self.prior_output.sigma_mix / settings.rho_scale_lower_output)
                - 1.0
            ),
            rho_upper=math.log(
                math.exp(self.prior_output.sigma_mix / settings.rho_scale_upper_output)
                - 1.0
            ),
        )

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
            if isinstance(x, PackedSequence):
                x, lens = pad_packed_sequence(x)
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

def _apply_permutation(tensor: Tensor, permutation: Tensor, dim: int = 1) -> Tensor:
    return tensor.index_select(dim, permutation)

class BayesRNNBase(Module):

    def __init__(
        self,
        mode: str,
        input_size: int,
        hidden_size: int,
        prior: object,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0,
        bidirectional: bool = False,
        mu_lower = -0.05,
        mu_upper = 0.05,
        rho_lower = -1,
        rho_upper = -4,
    ):
        super().__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.prior = prior
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.dropout_state = {}
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        if (
            not isinstance(dropout, numbers.Number)
            or not 0 <= dropout <= 1
            or isinstance(dropout, bool)
        ):
            raise ValueError(
                "dropout should be a number in range [0, 1] "
                "representing the probability of an element being "
                "zeroed"
            )
        if dropout > 0 and num_layers == 1:
            warnings.warn(
                "dropout option adds dropout after all but last "
                "recurrent layer, so non-zero dropout expects "
                "num_layers greater than 1, but got dropout={} and "
                "num_layers={}".format(dropout, num_layers)
            )

        if mode == "LSTM":
            gate_size = 4 * hidden_size
        elif mode == "GRU":
            gate_size = 3 * hidden_size
        else:
            gate_size = hidden_size

        self._flat_weights_names = []
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = (
                    input_size if layer == 0 else hidden_size * num_directions
                )

                w_ih_mu, w_ih_rho = get_bbb_variable(
                    (gate_size, layer_input_size),
                    mu_lower,
                    mu_upper,
                    rho_lower,
                    rho_upper,
                )
                
                w_hh_mu, w_hh_rho = get_bbb_variable(
                    (gate_size, hidden_size), mu_lower, mu_upper, rho_lower, rho_upper
                )

                b_ih_mu, b_ih_rho = get_bbb_variable(
                    (gate_size,), mu_lower, mu_upper, rho_lower, rho_upper
                )
                b_hh_mu, b_hh_rho = get_bbb_variable(
                    (gate_size,), mu_lower, mu_upper, rho_lower, rho_upper
                )

                if bias:
                    layer_params = (
                        w_ih_mu,
                        w_ih_rho,
                        w_hh_mu,
                        w_hh_rho,
                        b_ih_mu,
                        b_ih_rho,
                        b_hh_mu,
                        b_hh_rho
                    )
                else:
                    layer_params = (
                        w_ih_mu,
                        w_ih_rho,
                        w_hh_mu,
                        w_hh_rho,
                    )


                suffix = "_reverse" if direction == 1 else ""
                param_names = [
                    "weight_ih_mu_l{}{}",
                    "weight_ih_rho_l{}{}",
                    "weight_hh_mu_l{}{}",
                    "weight_hh_rho_l{}{}",
                ]

                if bias:
                    param_names += [
                        "bias_ih_mu_l{}{}",
                        "bias_ih_rho_l{}{}",
                        "bias_hh_mu_l{}{}",
                        "bias_hh_rho_l{}{}",
                    ]
                param_names = [x.format(layer, suffix) for x in param_names]

                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._flat_weights_names.extend(param_names)
                self._all_weights.append(param_names)

        self.flatten_parameters()
        self.kl = None

    def flatten_parameters(self) -> None:
        """Resets parameter data pointer so that they can use faster code paths.
        """
        self._data_ptrs = []
        self._param_buf_size = 0
        return

    def _apply(self, fn):
        ret = super(BayesRNNBase, self)._apply(fn)
        self.flatten_parameters()
        return ret

    def check_input(self, input: Tensor, batch_sizes: Optional[Tensor]) -> None:
        expected_input_dim = 2 if batch_sizes is not None else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.size(-1)))
    
    def get_expected_hidden_size(self, input: Tensor, batch_sizes: Optional[Tensor]) -> Tuple[int, int, int]:
        if batch_sizes is not None:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)
        num_directions = 2 if self.bidirectional else 1
       
        expected_hidden_size = (self.num_layers * num_directions,
                                mini_batch, self.hidden_size)
    
        return expected_hidden_size
    
    def check_hidden_size(self, hx: Tensor, expected_hidden_size: Tuple[int, int, int],
                          msg: str = 'Expected hidden size {}, got {}') -> None:
        if hx.size() != expected_hidden_size:
            raise RuntimeError(msg.format(expected_hidden_size, list(hx.size())))
        
    def check_forward_args(self, input: Tensor, hidden: Tensor, batch_sizes: Optional[Tensor]):
        self.check_input(input, batch_sizes)
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)

        self.check_hidden_size(hidden, expected_hidden_size)

    def permute_hidden(self, hx: Tensor, permutation: Optional[Tensor]):
        if permutation is None:
            return hx
        return _apply_permutation(hx, permutation)
    
    def extra_repr(self) -> str:
        s = "{input_size}, {hidden_size}"
        if self.num_layers != 1:
            s += ", num_layers={num_layers}"
        if self.bias is not True:
            s += ", bias={bias}"
        if self.batch_first is not False:
            s += ", batch_first={batch_first}"
        if self.dropout != 0:
            s += ", dropout={dropout}"
        if self.bidirectional is not False:
            s += ", bidirectional={bidirectional}"
        return s.format(**self.__dict__)


class BayesLSTM(BayesRNNBase):

    def __init__(self, *args, **kwargs):
        super().__init__("LSTM", *args, **kwargs)

    def get_expected_cell_size(self, input: Tensor, batch_sizes: Optional[Tensor]) -> Tuple[int, int, int]:
        if batch_sizes is not None:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)
        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions,
                                mini_batch, self.hidden_size)
        return expected_hidden_size

    def check_forward_args(self,  # type: ignore[override]
                        input: Tensor,
                        hidden: Tuple[Tensor, Tensor],
                        batch_sizes: Optional[Tensor],
                        ):
        self.check_input(input, batch_sizes)
        self.check_hidden_size(hidden[0], self.get_expected_hidden_size(input, batch_sizes),
                                'Expected hidden[0] size {}, got {}')
        self.check_hidden_size(hidden[1], self.get_expected_cell_size(input, batch_sizes),
                                'Expected hidden[1] size {}, got {}')
        
    
    def permute_hidden(self,  # type: ignore[override]
                    hx: Tuple[Tensor, Tensor],
                    permutation: Optional[Tensor]
                    ) -> Tuple[Tensor, Tensor]:
        if permutation is None:
            return hx
        return _apply_permutation(hx[0], permutation), _apply_permutation(hx[1], permutation)
    
    def forward(self, input, hx=None, mean_field_inference=False):
        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            batch_sizes = None
            assert (input.dim() in (2, 3)), f"LSTM: Expected input to be 2-D or 3-D but received {input.dim()}-D tensor"
            is_batched = input.dim() == 3
            batch_dim = 0 if self.batch_first else 1
            if not is_batched:
                input = input.unsqueeze(batch_dim)
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
    
            hx = torch.zeros(self.num_layers * num_directions,
                                max_batch_size, self.hidden_size,
                                dtype=input.dtype, device=input.device)
            
            hx = (hx, hx)

        # else:
        #     if batch_sizes is None:  # If not PackedSequence input.
        #         if is_batched:
        #             if (hx[0].dim() != 3 or hx[1].dim() != 3):
        #                 msg = ("For batched 3-D input, hx and cx should "
        #                         f"also be 3-D but got ({hx[0].dim()}-D, {hx[1].dim()}-D) tensors")
        #                 raise RuntimeError(msg)
        #         else:
        #             if hx[0].dim() != 2 or hx[1].dim() != 2:
        #                 msg = ("For unbatched 2-D input, hx and cx should "
        #                         f"also be 2-D but got ({hx[0].dim()}-D, {hx[1].dim()}-D) tensors")
        #                 raise RuntimeError(msg)
        #             hx = (hx[0].unsqueeze(1), hx[1].unsqueeze(1))

            # # Each batch of the hidden state should match the input sequence that
            # # the user believes he/she is passing in.
            # hx = self.permute_hidden(hx, sorted_indices)

        # todo: memory usage efficiency
        # has_flat_weights = (
        #     list(p.data.data_ptr() for p in self.parameters()) == self._data_ptrs
        # )
        # if has_flat_weights:
        #     first_data = next(self.parameters()).data
        #     assert first_data.storage().size() == self._param_buf_size
        #     flat_weight = first_data.new().set_(
        #         first_data.storage(), 0, torch.Size([self._param_buf_size])
        #     )
        # else:
        #     flat_weight = None
       
        self.check_forward_args(input, hx, batch_sizes)

        # Format weights for BBB
        all_weights = []
        num_layers = self.num_layers
        num_directions = 2 if self.bidirectional else 1
        self.kl = 0
        # Loop over layers
        for layer_idx in range(num_layers):
            # Loop over directions
            for direction in range(num_directions):

                suffix = "_reverse" if direction == 1 else ""

                w_ih_mean = getattr(self, f"weight_ih_mu_l{layer_idx}{suffix}")
                w_ih_rho = getattr(self, f"weight_ih_rho_l{layer_idx}{suffix}")
                w_ih_sigma = F.softplus(w_ih_rho) + 1e-5

                w_hh_mean = getattr(self, f"weight_hh_mu_l{layer_idx}{suffix}")
                w_hh_rho = getattr(self, f"weight_hh_rho_l{layer_idx}{suffix}")
                w_hh_sigma = F.softplus(w_hh_rho) + 1e-5

                if mean_field_inference:
                    weight_ih = w_ih_mean
                    weight_hh = w_hh_mean
                
                else:
                    # Sample weights from normal distribution
                    eps_ih = w_ih_mean.data.new(w_ih_mean.size()).normal_(0.0, 1.0)
                    weight_ih = w_ih_mean + eps_ih * w_ih_sigma

                    eps_hh = w_hh_mean.data.new(w_hh_mean.size()).normal_(0.0, 1.0)
                    weight_hh = w_hh_mean + eps_hh * w_hh_sigma

                all_weights.extend([weight_ih, weight_hh])

                # Compute KL divergence
                self.kl += compute_KL(weight_ih, w_ih_mean, w_ih_sigma, self.prior)
                self.kl += compute_KL(weight_hh, w_hh_mean, w_hh_sigma, self.prior)

                if self.bias:

                    b_ih_mean = getattr(self, f"bias_ih_mu_l{layer_idx}{suffix}")
                    b_ih_rho = getattr(self, f"bias_ih_rho_l{layer_idx}{suffix}")
                    b_ih_sigma = F.softplus(b_ih_rho) + 1e-5

                    b_hh_mean = getattr(self, f"bias_hh_mu_l{layer_idx}{suffix}")
                    b_hh_rho = getattr(self, f"bias_hh_rho_l{layer_idx}{suffix}")
                    b_hh_sigma = F.softplus(b_hh_rho) + 1e-5

                    if mean_field_inference:
                        bias_ih = b_ih_mean
                        bias_hh = b_hh_mean
               
                    else:
                        eps_ih = b_ih_mean.data.new(b_ih_mean.size()).normal_(0.0, 1.0)
                        bias_ih = b_ih_mean + eps_ih * b_ih_sigma

                        eps_hh = b_hh_mean.data.new(b_hh_mean.size()).normal_(0.0, 1.0)
                        bias_hh = b_hh_mean + eps_hh * b_hh_sigma

                    all_weights.extend([bias_ih, bias_hh])
                
                    self.kl += compute_KL(bias_ih, b_ih_mean, b_ih_sigma, self.prior)
                    self.kl += compute_KL(bias_hh, b_hh_mean, b_hh_sigma, self.prior)

        if batch_sizes is None:
            result = _VF.lstm(input, hx, all_weights, self.bias, self.num_layers,
                            self.dropout, self.training, self.bidirectional, self.batch_first)
        else:
            result = _VF.lstm(input, batch_sizes, hx, all_weights, self.bias,
                            self.num_layers, self.dropout, self.training, self.bidirectional)
        output = result[0]
        hidden = result[1:]

        if is_packed:
            output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
            return output_packed, self.permute_hidden(hidden, unsorted_indices)
        else:
            if not is_batched:
                output = output.squeeze(batch_dim)
                hidden = (hidden[0].squeeze(1), hidden[1].squeeze(1))
            return output, self.permute_hidden(hidden, unsorted_indices)


class BayesGRU(BayesRNNBase):

    def __init__(self, *args, **kwargs):
        super().__init__("GRU", *args, **kwargs)


class BayesLinear(Module):

    def __init__(
        self, in_features, out_features, prior, mu_lower, mu_upper, rho_lower, rho_upper
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior = prior

        mu, rho = get_bbb_variable(
            (out_features, in_features), mu_lower, mu_upper, rho_lower, rho_upper
        )

        bias = Parameter(torch.Tensor(out_features))
        bias.data.fill_(0.0)

        self.mu = mu
        self.rho = rho
        self.bias = bias
        self.kl = None

    def forward(self, input, mean_field_inference=False):

        # Sample weight
        mean = self.mu
        sigma = F.softplus(self.rho) + 1e-5

        if mean_field_inference:
            weights = mean
        else:
            # Sample weights from normal distribution
            # This way of creating the epsilon variable is faster than
            # from numpy or torch.randn or FloatTensor.normal_ when mean is already
            # on the GPU
            eps = mean.data.new(mean.size()).normal_(0.0, 1.0)
            weights = mean + eps * sigma

        logits = F.linear(input, weights, self.bias)

        # Compute KL divergence
        self.kl = compute_KL(weights, mean, sigma, self.prior)

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


class BayesBiasLinear(Module):

    def __init__(
        self, in_features, out_features, prior, mu_lower, mu_upper, rho_lower, rho_upper
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior = prior

        self.W_mu, self.W_rho = get_bbb_variable(
            (out_features, in_features), mu_lower, mu_upper, rho_lower, rho_upper
        )

        self.b_mu, self.b_rho = get_bbb_variable(
            (out_features,), mu_lower, mu_upper, rho_lower, rho_upper
        )

    def forward(self, X, mean_field_inference=False):
        # Sample weight
        W_mean = self.W_mu
        W_sigma = F.softplus(self.W_rho) + 1e-5

        b_mean = self.b_mu
        b_sigma = F.softplus(self.b_rho) + 1e-5

        if mean_field_inference:
            weights = W_mean
            biases = b_mean
        else:
            # Sample weights from normal distribution
            # This way of creating the epsilon variable is faster than
            # from numpy or torch.randn or FloatTensor.normal_ when mean is already
            # on the GPU
            W_eps = W_mean.data.new(W_mean.size()).normal_(0.0, 1.0)
            weights = W_mean + W_eps * W_sigma

            b_eps = b_mean.data.new(b_mean.size()).normal_(0.0, 1.0)
            biases = b_mean + b_eps * b_sigma

        logits = F.linear(X, weights, biases)

        # Compute KL divergence
        self.kl = compute_KL(weights, W_mean, W_sigma, self.prior)
        self.kl += compute_KL(biases, b_mean, b_sigma, self.prior)

        return logits


class BayesEmbedding(Module):

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

    def forward(self, input, mean_field_inference=False):

        # Sample weight
        mean = self.mu
        sigma = F.softplus(self.rho) + 1E-5

        if mean_field_inference:
            weights = mean
        else:
            # This way of creating the epsilon variable is faster than
            # from numpy or torch.randn or FloatTensor.normal_ when mean is already
            # on the GPU
            eps = mean.data.new(mean.size()).normal_(0., 1.)
            weights = mean + eps * sigma

        # Compute KL divergence
        self.kl = compute_KL(weights, mean, sigma, self.prior)

        after_embed = F.embedding(
            input,
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

    mu = Parameter(torch.Tensor(*shape))
    rho = Parameter(torch.Tensor(*shape))

    # Initialize
    mu.data.uniform_(mu_lower, mu_upper)
    rho.data.uniform_(rho_lower, rho_upper)

    return mu, rho


def logsumexp(x, dim):
    """Logsumexp trick to avoid overflow in a log of sum of exponential expression

    Args:
        x (Tensor): the input on which to compute the log of sum of exponential

    Returns:
        logsum (Tensor): the computed log of sum of exponential
    """

    assert x.dim() == 2
    x_max, x_max_idx = x.max(dim=dim, keepdim=True)
    logsum = x_max + torch.log((x - x_max).exp().sum(dim=dim, keepdim=True))
    return logsum


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
