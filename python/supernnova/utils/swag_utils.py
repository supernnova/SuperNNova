# This file is based on code from:
# - https://github.com/pytorch/pytorch/blob/v2.0.1/torch/optim/swa_utils.py
# - https://github.com/pytorch/pytorch/blob/main/torch/optim/swa_utils.py
# It implements Stochastic Weight Averaging-Gaussian (SWAG) from the paper
# "A Simple Baseline for Bayesian Uncertainty in Deep Learning" by Wesley Maddox et al.
# Some functions are also adapted from https://github.com/wjmaddox/swa_gaussian/tree/master.

# Author: J. Hu

# Relevant license content follows.

# From PyTorch:

# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

# From Caffe2:

# Copyright (c) 2016-present, Facebook Inc. All rights reserved.

# All contributions by Facebook:
# Copyright (c) 2016 Facebook Inc.

# All contributions by Google:
# Copyright (c) 2015 Google Inc.
# All rights reserved.

# All contributions by Yangqing Jia:
# Copyright (c) 2015 Yangqing Jia
# All rights reserved.

# All contributions by Kakao Brain:
# Copyright 2019-2020 Kakao Brain

# All contributions by Cruise LLC:
# Copyright (c) 2022 Cruise LLC.
# All rights reserved.

# All contributions from Caffe:
# Copyright(c) 2013, 2014, 2015, the respective contributors
# All rights reserved.

# All other contributions:
# Copyright(c) 2015, 2016 the respective contributors
# All rights reserved.

# Caffe2 uses a copyright model similar to Caffe: each contributor holds
# copyright over their contributions to Caffe2. The project versioning records
# all such contribution and copyright details. If a contributor wants to further
# mark their specific copyright on a particular contribution, they should
# indicate their copyright solely in the commit message of the change when it is
# committed.

# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.

# 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
#    and IDIAP Research Institute nor the names of its contributors may be
#    used to endorse or promote products derived from this software without
#    specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# BSD 2-Clause License

# Copyright (c) 2019, Wesley Maddox, Timur Garipov, Pavel Izmailov,  Dmitry Vetrov, Andrew Gordon Wilson
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
        self._init_swag_params()

    def _init_swag_params(self):
        for i, p in enumerate(self.module.parameters()):
            sec_moment_name = f"sec_moment_{i}"
            sec_moment_tensor = torch.zeros_like(p)
            self.register_buffer(sec_moment_name, sec_moment_tensor)
            dev_name = f"dev_{i}"
            dev_tensor = p.new_empty((p.numel(), 0)).zero_()
            self.register_buffer(dev_name, dev_tensor)

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

        for i, (p_swa, p_model) in enumerate(zip(self_param, model_param)):
            device = p_swa.device
            p_swa_ = p_swa.detach()
            p_model_ = p_model.detach().to(device)
            if self.n_averaged == 0:
                p_swa_.copy_(p_model_)
                setattr(self, f"sec_moment_{i}", p_model_ * p_model_)

            else:
                p_swa_.copy_(self.avg_fn(p_swa_, p_model_, self.n_averaged.to(device)))

                sec_moment_value = second_moment_update(
                    getattr(self, f"sec_moment_{i}"),
                    p_model_,
                    self.n_averaged.to(device),
                )
                setattr(self, f"sec_moment_{i}", sec_moment_value)

                dev = (p_model_ - p_swa_).view(-1, 1)
                dev_value = torch.cat((getattr(self, f"dev_{i}"), dev), dim=1)

                setattr(self, f"dev_{i}", dev_value)

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
        num_param = len(list(self.module.parameters()))
        sec_moment_list = [getattr(self, f"sec_moment_{i}") for i in range(num_param)]
        sec_moment = flatten(sec_moment_list)

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
            dev_list = [getattr(self, f"dev_{i}") for i in range(num_param)]
            dev = torch.cat(dev_list, dim=0)
            cov_sample = dev.matmul(
                dev.new_empty((dev.size(1),), requires_grad=False).normal_()
            )

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
