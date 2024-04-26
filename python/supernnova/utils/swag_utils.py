# This file is originally from https://github.com/pytorch/pytorch/blob/v2.0.1/torch/optim/swa_utils.py
# and https://github.com/pytorch/pytorch/blob/main/torch/optim/swa_utils.py with modifications.
# with reference from https://github.com/wjmaddox/swa_gaussian/tree/master

import itertools
from copy import deepcopy
from typing import Union

import torch
from torch import Tensor
from torch.nn import Module


@torch.no_grad()
def swa_update(
    averaged_param: Tensor, current_param: Tensor, num_averaged: Union[Tensor, int]
):
    return averaged_param + (current_param - averaged_param) / (num_averaged + 1)


@torch.no_grad()
def second_moment_update(
    averaged_second_moment: Tensor,
    current_param: Tensor,
    num_averaged: Union[Tensor, int],
):
    return averaged_second_moment + (
        current_param * current_param - averaged_second_moment
    ) / (num_averaged + 1)


def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)


def unflatten_like(vector, likeTensorList):
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    #    shaped like likeTensorList
    outList = []
    i = 0
    for tensor in likeTensorList:
        n = tensor.numel()
        outList.append(vector[:, i : i + n].view(tensor.shape))
        i += n
    return outList


class SwagModel(Module):
    r"""Implements averaged model for Stochastic Weight Averaging (SWA).

    Stochastic Weight Averaging was proposed in `Averaging Weights Leads to
    Wider Optima and Better Generalization`_ by Pavel Izmailov, Dmitrii
    Podoprikhin, Timur Garipov, Dmitry Vetrov and Andrew Gordon Wilson
    (UAI 2018).

    AveragedModel class creates a copy of the provided module :attr:`model`
    on the device :attr:`device` and allows to compute running averages of the
    parameters of the :attr:`model`.

    Args:
        model (torch.nn.Module): model to use with SWA
        device (torch.device, optional): if provided, the averaged model will be
            stored on the :attr:`device`
        avg_fn (function, optional): the averaging function used to update
            parameters; the function must take in the current value of the
            :class:`AveragedModel` parameter, the current value of :attr:`model`
            parameter and the number of models already averaged; if None,
            equally weighted average is used (default: None)
        use_buffers (bool): if ``True``, it will compute running averages for
            both the parameters and the buffers of the model. (default: ``False``)

    Example:
        >>> # xdoctest: +SKIP("undefined variables")
        >>> loader, optimizer, model, loss_fn = ...
        >>> swa_model = torch.optim.swa_utils.AveragedModel(model)
        >>> scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
        >>>                                     T_max=300)
        >>> swa_start = 160
        >>> swa_scheduler = SWALR(optimizer, swa_lr=0.05)
        >>> for i in range(300):
        >>>      for input, target in loader:
        >>>          optimizer.zero_grad()
        >>>          loss_fn(model(input), target).backward()
        >>>          optimizer.step()
        >>>      if i > swa_start:
        >>>          swa_model.update_parameters(model)
        >>>          swa_scheduler.step()
        >>>      else:
        >>>          scheduler.step()
        >>>
        >>> # Update bn statistics for the swa_model at the end
        >>> torch.optim.swa_utils.update_bn(loader, swa_model)

    You can also use custom averaging functions with `avg_fn` parameter.
    If no averaging function is provided, the default is to compute
    equally-weighted average of the weights.

    Example:
        >>> # xdoctest: +SKIP("undefined variables")
        >>> # Compute exponential moving averages of the weights and buffers
        >>> ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: (
        ...                 0.1 * averaged_model_parameter + 0.9 * model_parameter)
        >>> swa_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg, use_buffers=True)

    .. note::
        When using SWA with models containing Batch Normalization you may
        need to update the activation statistics for Batch Normalization.
        This can be done either by using the :meth:`torch.optim.swa_utils.update_bn`
        or by setting :attr:`use_buffers` to `True`. The first approach updates the
        statistics in a post-training step by passing data through the model. The
        second does it during the parameter update phase by averaging all buffers.
        Empirical evidence has shown that updating the statistics in normalization
        layers increases accuracy, but you may wish to empirically test which
        approach yields the best results in your problem.

    .. note::
        :attr:`avg_fn` is not saved in the :meth:`state_dict` of the model.

    .. note::
        When :meth:`update_parameters` is called for the first time (i.e.
        :attr:`n_averaged` is `0`) the parameters of `model` are copied
        to the parameters of :class:`AveragedModel`. For every subsequent
        call of :meth:`update_parameters` the function `avg_fn` is used
        to update the parameters.

    .. _Averaging Weights Leads to Wider Optima and Better Generalization:
        https://arxiv.org/abs/1803.05407
    .. _There Are Many Consistent Explanations of Unlabeled Data: Why You Should
        Average:
        https://arxiv.org/abs/1806.05594
    .. _SWALP: Stochastic Weight Averaging in Low-Precision Training:
        https://arxiv.org/abs/1904.11943
    .. _Stochastic Weight Averaging in Parallel: Large-Batch Training That
        Generalizes Well:
        https://arxiv.org/abs/2001.02312
    """

    def __init__(
        self, model: Module, device=None, avg_fn=swa_update, use_buffers=False
    ):
        super().__init__()
        self.module = deepcopy(model)
        if device is not None:
            self.module = self.module.to(device)
        self.register_buffer(
            "n_averaged", torch.tensor(0, dtype=torch.long, device=device)
        )
        self.avg_fn = avg_fn
        self.use_buffers = use_buffers

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def update_parameters(self, model: Module):
        self_param = (
            itertools.chain(self.module.parameters(), self.module.buffers())
            if self.use_buffers
            else self.parameters()
        )
        model_param = (
            itertools.chain(model.parameters(), model.buffers())
            if self.use_buffers
            else model.parameters()
        )

        if not hasattr(self, "sec_moment_collect"):
            self.sec_moment_collect = [
                torch.zeros_like(p) for p in self.module.parameters()
            ]

        if not hasattr(self, "dev_collect"):
            self.dev_collect = [
                p.new_empty((p.numel(), 0)).zero_() for p in self.module.parameters()
            ]

        for i, (p_swa, p_model) in enumerate(zip(self_param, model_param)):
            device = p_swa.device
            p_swa_ = p_swa.detach()
            p_model_ = p_model.detach().to(device)
            if self.n_averaged == 0:
                p_swa_.copy_(p_model_)
                self.sec_moment_collect[i] = p_model_ * p_model_
            else:
                p_swa_.copy_(
                    self.avg_fn(p_swa.detach(), p_model_, self.n_averaged.to(device))
                )
                self.sec_moment_collect[i] = second_moment_update(
                    self.sec_moment_collect[i], p_model_, self.n_averaged.to(device)
                )
                dev = (p_model_ - p_swa_).view(-1, 1)
                self.dev_collect[i] = torch.cat((self.dev_collect[i], dev), dim=1)

        if not self.use_buffers:
            # If not apply running averages to the buffers,
            # keep the buffers in sync with the source model.
            for b_swa, b_model in zip(self.module.buffers(), model.buffers()):
                b_swa.detach().copy_(b_model.detach().to(device))
        self.n_averaged += 1

    def sample(self, scale=0.5, cov=True, var_clamp=1e-30):
        """sampling of SWAG model; should be used when the training is finished"""
        self.sample_model = deepcopy(self.module)

        scale_sqrt = scale**0.5

        # get the mean, second moment
        mean_list = [p for p in self.module.parameters()]
        mean = flatten(mean_list)
        sec_moment = flatten(self.sec_moment_collect)

        # draw diagonal variance sample
        var = torch.clamp(sec_moment - mean**2, var_clamp)
        var_sample = var.sqrt() * torch.randn_like(var, requires_grad=False)

        # if covariance draw low rank sample
        if cov:
            dev = torch.cat(self.dev_collect, dim=0)

            cov_sample = dev.matmul(
                dev.new_empty((dev.size(1),), requires_grad=False).normal_()
            )
            cov_sample /= (self.n_averaged - 1) ** 0.5

            rand_sample = var_sample + cov_sample
        else:
            rand_sample = var_sample

        # update sample with mean and scale
        sample = mean + scale_sqrt * rand_sample
        sample = sample.unsqueeze(0)

        # unflatten new sample like the mean sample
        samples_list = unflatten_like(sample, mean_list)

        # update sample model
        for p_sample, sample_tensor in zip(
            self.sample_model.parameters(), samples_list
        ):
            p_sample_ = p_sample.detach()
            p_sample_.copy_(sample_tensor)
