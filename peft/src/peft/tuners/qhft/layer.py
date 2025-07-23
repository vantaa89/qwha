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
from .hadamard import wht, iwht


class QHFTLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("qhft_spectrum",)
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("qhft_n_frequency", "qhft_scaling", "qhft_random_loc_seed", "qhft_indices")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer

        # Initialize as empty dicts (same as original)
        self.qhft_n_frequency = {}
        self.qhft_scaling = {}
        self.qhft_spectrum = nn.ParameterDict({})
        self.qhft_indices = {}
        self.qhft_random_loc_seed = {}

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

    def update_layer(self, adapter_name, n_frequency, scaling, init_weights, random_loc_seed):
        if n_frequency <= 0:
            raise ValueError(f"`n_frequency` should be a positive integer value but the value passed is {n_frequency}")
        if n_frequency > self.in_features * self.out_features:
            raise ValueError(
                f"`n_frequency` should be less than or equal to the product of the input and output dimensions "
                f"but the value passed is {n_frequency} and the product is {self.in_features * self.out_features}"
            )

        # Register as buffers/parameters for automatic state_dict inclusion
        self.register_buffer(f"qhft_n_frequency_{adapter_name}", torch.tensor(n_frequency, dtype=torch.long))
        self.register_buffer(f"qhft_random_loc_seed_{adapter_name}", torch.tensor(random_loc_seed, dtype=torch.long))
        self.register_parameter(f"qhft_scaling_{adapter_name}", nn.Parameter(torch.tensor(scaling, dtype=torch.float32), requires_grad=False))

        # Generate and register indices
        indices = torch.randperm(
            self.out_features * self.in_features,
            generator=torch.Generator().manual_seed(random_loc_seed),
        )[:n_frequency]
        indices = torch.stack([indices // self.in_features, indices % self.in_features], dim=0)
        self.register_buffer(f"qhft_indices_{adapter_name}", indices)

        # Update the dict references (for backward compatibility)
        self.qhft_n_frequency[adapter_name] = n_frequency
        self.qhft_random_loc_seed[adapter_name] = random_loc_seed
        self.qhft_scaling[adapter_name] = scaling
        self.qhft_indices[adapter_name] = indices

        # Actual trainable parameters (already handled by ParameterDict)
        self.qhft_spectrum[adapter_name] = nn.Parameter(torch.randn(n_frequency), requires_grad=True)

        if init_weights:
            self.reset_qhft_parameters(adapter_name)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def update_indices(self, adapter_name, new_indices):
        """Update the indices of the adapter and modify the registered buffer"""
        if adapter_name not in self.qhft_indices:
            raise ValueError(f"Adapter {adapter_name} not found")

        # Update the dictionary
        self.qhft_indices[adapter_name] = new_indices

        # Update the buffer
        buffer_name = f"qhft_indices_{adapter_name}"
        if hasattr(self, buffer_name):
            setattr(self, buffer_name, new_indices)
        else:
            self.register_buffer(buffer_name, new_indices)


    @torch.no_grad()
    def reset_qhft_parameters(self, adapter_name):
        if adapter_name in self.qhft_spectrum.keys():
            nn.init.zeros_(self.qhft_spectrum[adapter_name])

    def get_delta_weight(self, adapter) -> torch.Tensor:
        spectrum = self.qhft_spectrum[adapter]
        indices = self.qhft_indices[adapter]
        dense_spectrum = torch.zeros(self.out_features, self.in_features, device=spectrum.device, dtype=spectrum.dtype)
        dense_spectrum[indices[0, :], indices[1, :]] = spectrum
        # iwht employs 1/N scaling, while forward wht does not scale
        delta_weight = iwht(
            dense_spectrum) * self.qhft_scaling[adapter] / math.sqrt(self.out_features)
        return delta_weight

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Override to handle loading of registered parameters back into dicts"""
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        # Rebuild dicts from loaded registered parameters/buffers
        self._rebuild_dicts_from_registered_params()

    def _rebuild_dicts_from_registered_params(self):
        """Rebuild the dict properties from registered parameters/buffers"""
        # Clear existing dicts
        self.qhft_n_frequency.clear()
        self.qhft_scaling.clear()
        self.qhft_indices.clear()
        self.qhft_random_loc_seed.clear()

        # Rebuild from registered parameters
        for name, param in self.named_parameters():
            if name.startswith("qhft_scaling_"):
                adapter_name = name.replace("qhft_scaling_", "")
                self.qhft_scaling[adapter_name] = param.item()

        # Rebuild from registered buffers
        for name, buffer in self.named_buffers():
            if name.startswith("qhft_n_frequency_"):
                adapter_name = name.replace("qhft_n_frequency_", "")
                self.qhft_n_frequency[adapter_name] = buffer.item()
            elif name.startswith("qhft_random_loc_seed_"):
                adapter_name = name.replace("qhft_random_loc_seed_", "")
                self.qhft_random_loc_seed[adapter_name] = buffer.item()
            elif name.startswith("qhft_indices_"):
                adapter_name = name.replace("qhft_indices_", "")
                self.qhft_indices[adapter_name] = buffer


class QHFTLinear(nn.Module, QHFTLayer):
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        n_frequency: int = 1000,
        scaling: float = 150.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        init_weights: Union[bool, str] = False,
        random_loc_seed: int = 777,
        **kwargs,
    ) -> None:
        super().__init__()
        QHFTLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, n_frequency, scaling, init_weights, random_loc_seed)

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
            if active_adapter in self.qhft_spectrum.keys():
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
            if active_adapter in self.qhft_spectrum.keys():
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
            else:
                base_w = self.base_layer.weight
            for active_adapter in self.active_adapters:
                if active_adapter not in self.qhft_spectrum.keys():
                    continue
                delta_w = self.get_delta_weight(active_adapter).to(x.dtype)
            result = F.linear(x, base_w + delta_w)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "qhft." + rep
