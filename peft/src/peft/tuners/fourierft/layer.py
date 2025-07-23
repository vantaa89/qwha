# Copyright 2024-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge

import math
import os
import sys
sys.path.append(os.path.dirname(__file__))  # modification by hyesung
from hadamard import wht, iwht


class FourierFTLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("fourierft_spectrum",)
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("fourierft_n_frequency", "fourierft_scaling", "fourierft_random_loc_seed", "fourierft_indices")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.fourierft_n_frequency = {}
        self.fourierft_scaling = {}
        self.fourierft_spectrum = nn.ParameterDict({})
        self.fourierft_indices = {}
        self.fourierft_random_loc_seed = {}
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            self.in_features, self.out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            self.in_features, self.out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        elif hasattr(base_layer, "qweight"):  # QuantLinear
            self.in_features, self.out_features = base_layer.in_features, base_layer.out_features
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

    def update_layer(self, adapter_name, n_frequency, scaling, init_weights, random_loc_seed, channel_wise, cosine_transform, hadamard_transform):
        if n_frequency <= 0:
            raise ValueError(f"`n_frequency` should be a positive integer value but the value passed is {n_frequency}")
        if n_frequency > self.in_features * self.out_features:
            raise ValueError(
                f"`n_frequency` should be less than or equal to the product of the input and output dimensions "
                f"but the value passed is {n_frequency} and the product is {self.in_features * self.out_features}"
            )
        self.fourierft_n_frequency[adapter_name] = n_frequency
        self.channel_wise = channel_wise
        self.cosine_transform = cosine_transform
        self.hadamard_transform = hadamard_transform
        self.fourierft_random_loc_seed[adapter_name] = random_loc_seed
        self.fourierft_indices[adapter_name] = torch.randperm(
            self.out_features * self.in_features,
            generator=torch.Generator().manual_seed(self.fourierft_random_loc_seed[adapter_name]),
        )[:n_frequency]
        self.fourierft_indices[adapter_name] = torch.stack(
            [self.fourierft_indices[adapter_name] // self.in_features, self.fourierft_indices[adapter_name] % self.in_features], dim=0
        )
        self.fourierft_scaling[adapter_name] = scaling
        # Actual trainable parameters
        self.fourierft_spectrum[adapter_name] = nn.Parameter(torch.randn(n_frequency), requires_grad=True)

        if init_weights:
            self.reset_fourier_parameters(adapter_name)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    @torch.no_grad()
    def reset_fourier_parameters(self, adapter_name):
        if adapter_name in self.fourierft_spectrum.keys():
            # for qhft**
            nn.init.zeros_(self.fourierft_spectrum[adapter_name])
            # for ssh
            # nn.init.kaiming_uniform_(self.fourierft_spectrum[adapter_name])

    def get_delta_weight(self, adapter) -> torch.Tensor:
        spectrum = self.fourierft_spectrum[adapter]
        indices = self.fourierft_indices[adapter]# .to(spectrum.device)
        dense_spectrum = torch.zeros(self.out_features, self.in_features, device=spectrum.device, dtype=spectrum.dtype)
        dense_spectrum[indices[0, :], indices[1, :]] = spectrum
        if self.cosine_transform:   # Use DCT instead of DFT
            if self.channel_wise:
                delta_weight = idct1(
                    dense_spectrum) * self.fourierft_scaling[adapter] / math.sqrt(self.out_features)
            else:
                delta_weight = idct1_2d(dense_spectrum) * self.fourierft_scaling[adapter]
        elif self.hadamard_transform:
            # iwht employs 1/N scaling, while forward wht does not scale
            delta_weight = iwht(
                dense_spectrum) * self.fourierft_scaling[adapter] / math.sqrt(self.out_features)
        else:  # Normal FourierFT using DFT
            # FIXME: ifft employs 1/N scaling, while forward fft does not scale.
            # dividing by math.sqrt(self.out_features) is not consistent with ifft2
            if self.channel_wise:
                delta_weight = torch.fft.ifft(
                    dense_spectrum, dim=1).real * self.fourierft_scaling[adapter] / math.sqrt(self.out_features)
            else:
                delta_weight = torch.fft.ifft2(
                    dense_spectrum).real * self.fourierft_scaling[adapter]
        return delta_weight


class FourierFTLinear(nn.Module, FourierFTLayer):
    # FourierFT implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        n_frequency: int = 1000,
        scaling: float = 150.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        init_weights: Union[bool, str] = False,
        random_loc_seed: int = 777,
        channel_wise: bool = False,
        cosine_transform: bool = False,
        hadamard_transform: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        FourierFTLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, n_frequency, scaling, init_weights,
                          random_loc_seed, channel_wise, cosine_transform, hadamard_transform)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.fourierft_spectrum.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.fourierft_spectrum.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        return super().get_delta_weight(adapter)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            if hasattr(self.base_layer, "qweight"):  # QuantLinear
                base_w = self.base_layer.dequantize_weight().T
                # from auto_gptq.nn_modules.triton_utils.dequant import dequant248 as dequant
                # base_w = dequant(self.base_layer.qweight,
                               # self.base_layer.scales,
                               # self.base_layer.qzeros,
                               # self.base_layer.g_idx,
                               # self.base_layer.bits,
                               # self.base_layer.maxq)
            else:
                base_w = self.base_layer.weight
            for active_adapter in self.active_adapters:
                if active_adapter not in self.fourierft_spectrum.keys():
                    continue
                delta_w = self.get_delta_weight(active_adapter).to(x.dtype)
            result = F.linear(x, base_w + delta_w)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "fourierft." + rep


# from https://github.com/zh217/torch-dct/blob/master/torch_dct/_dct.py

def report_cache(f):
    def decorated_f(x):
        # print(f"Memory allocated {torch.cuda.memory_allocated() / 1024**3:.2f}GB", end=" ")
        ret = f(x)
        # torch.cuda.empty_cache()
        # num_devices = torch.cuda.device_count()
        # for device_idx in range(num_devices):
            # torch.backends.cuda.cufft_plan_cache[device_idx].clear()
        # print(f"Memory allocated {torch.cuda.memory_allocated() / 1024**3:.2f}GB", end="\n")
        return ret
    return decorated_f

try:
    # PyTorch 1.7.0 and newer versions
    import torch.fft

    @report_cache
    def dct1_rfft_impl(x):
        return torch.view_as_real(torch.fft.rfft(x, dim=1))

    @report_cache
    def dct_fft_impl(v):
        return torch.view_as_real(torch.fft.fft(v, dim=1))

    @report_cache
    def idct_irfft_impl(V):
        return torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
except ImportError:
    # PyTorch 1.6.0 and older versions
    @report_cache
    def dct1_rfft_impl(x):
        return torch.rfft(x, 1)

    @report_cache
    def dct_fft_impl(v):
        return torch.rfft(v, 1, onesided=False)

    @report_cache
    def idct_irfft_impl(V):
        return torch.irfft(V, 1, onesided=False)


def dct1(x):
    """
    Discrete Cosine Transform, Type I

    :param x: the input signal
    :return: the DCT-I of the signal over the last dimension
    """
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])
    x = torch.cat([x, x.flip([1])[:, 1:-1]], dim=1)

    return dct1_rfft_impl(x)[:, :, 0].view(*x_shape) / 2


def idct1(X):
    """
    The inverse of DCT-I, which is just a scaled DCT-I

    Our definition if idct1 is such that idct1(dct1(x)) == x

    :param X: the input signal
    :return: the inverse DCT-I of the signal over the last dimension
    """
    n = X.shape[-1]
    return dct1(X) * 2 / (n - 1)


def idct1_2d(X, norm=None):
    """
    The inverse to 2D DCT-I, which is a scaled Discrete Cosine Transform, Type I

    Our definition of idct is that idct_2d(dct_2d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct1(X)
    x2 = idct1(x1.transpose(-1, -2))
    return x2.transpose(-1, -2)

def dht(x):
    """ Compute the DHT for a sequence x of length n using the FFT.
    """
    X = torch.fft.fft(x)
    return X.real - X.imag

def idht(X):
    """ Compute the IDHT for a sequence x of length n using the FFT.

    Since the DHT is involutory, IDHT(x) = 1/n DHT(H) = 1/n DHT(DHT(x))
    """
    n = X.shape[-1]
    x = dht(X)
    return x / n
