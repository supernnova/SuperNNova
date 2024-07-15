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
    r"""Implements averaged model for Stochastic Weight Averaging Gaussian (SWAG).

    SWAG was proposed in `A Simple Baseline for Bayesian Uncertainty
    in Deep Learning` by Wesley J. Maddox, Timur Garipov, Pavel Izmailov,
    Dmitry Vetrov and Andrew Gordon Wilson in 2019.

    SwagModel is a modified version of `torch.optim.swa_utils.AveragedModel` (`pytorch` v2.0.1), which
    implements Stochastic Weight Averaging (SWA). This modification is inspired by
    the original implementation of SWAG available at: https://github.com/wjmaddox/swa_gaussian.

    SwagModel class creates a copy of the provided module :attr:`model`
    on the device :attr:`device` and allows to compute running averages of the
    parameters of the :attr:`model`.

    Args:
        model (torch.nn.Module): model to use with SWAG
        device (torch.device, optional): if provided, the averaged model will be
            stored on the :attr:`device` (default: ``None``)
        avg_fn (function): the averaging function used to update
            parameters; the function must take in the current value of the
            :class:`SwagModel` parameter, the current value of :attr:`model`
            parameter and the number of models already averaged; if None,
            equally weighted average is used (default: ``swa_update``)
        use_buffers (bool): if ``True``, it will compute running averages for
            both the parameters and the buffers of the model. (default: ``False``)

    .. note::
        :attr:`avg_fn` is not saved in the :meth:`state_dict` of the model.

    .. note::
        When :meth:`update_parameters` is called for the first time (i.e.
        :attr:`n_averaged` is `0`) the parameters of `model` are copied
        to the parameters of :class:`SwagModel`. For every subsequent
        call of :meth:`update_parameters` the function `avg_fn` is used
        to update the parameters.
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
        """sampling of SWAG model"""
        sample_model = deepcopy(self.module)

        scale_sqrt = scale**0.5

        # get the mean, second moment
        mean_list = [p for p in self.module.parameters()]
        mean = flatten(mean_list)
        if torch.isnan(mean).any():
            print("mean contains NAN value")
            breakpoint()
        sec_moment = flatten(self.sec_moment_collect)

        # draw diagonal variance sample
        var = torch.clamp(sec_moment - mean**2, var_clamp)
        with torch.no_grad():
            var_sample = var.sqrt() * torch.randn_like(var)
        # temp: just for debugging
        self.var_sample = var_sample

        if torch.isnan(var_sample).any():
            print("var_sample contain NAN value")
            breakpoint()
        # if covariance draw low rank sample
        if cov:
            dev = torch.cat(self.dev_collect, dim=0)

            cov_sample = dev.matmul(
                dev.new_empty((dev.size(1),), requires_grad=False).normal_()
            )
            # temp: just for debugging
            self.cov_sample = cov_sample

            # we start deviation collect when self.n_averaged = 1 instead of 0
            # so K = self.n_averaged -1
            cov_sample /= (self.n_averaged - 2) ** 0.5

            if torch.isnan(cov_sample).any():
                print("cov_sample contains NAN value")
                breakpoint()
            rand_sample = var_sample + cov_sample
        else:
            rand_sample = var_sample

        # update sample with mean and scale
        sample = mean + scale_sqrt * rand_sample
        sample = sample.unsqueeze(0)

        # unflatten new sample like the mean sample
        samples_list = unflatten_like(sample, mean_list)

        # update sample model
        for p_sample, sample_tensor in zip(sample_model.parameters(), samples_list):
            p_sample_ = p_sample.detach()
            p_sample_.copy_(sample_tensor)

        return sample_model
