import gc
import os
import sys
import copy
import warnings
from tqdm import tqdm
from typing import Any, List, Optional, Union
from argparse import ArgumentParser, BooleanOptionalAction

from math import sqrt
from torch import nn
import torch
import torch.linalg
import torch.autograd.profiler as profiler
import numpy as np
import random

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
from peft import PeftModel, QHFTConfig, TaskType, get_peft_model, LoraConfig

from fast_hadamard_transform import hadamard_transform
from hadamard import wht, iwht
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import QHFT_CACHE_PATH, get_quantized_peft_model, rtn_quantize

ALPHA = 1.0


def get_initialized_model(
    model_id: str,
    bits: int = 4,
    peft_config: QHFTConfig | LoraConfig | None = None,
    dataset_id: str = "wikitext2",
    quant_method: str = "GPTQ",
    lora_rank: int = 64,
    group_size: int = 64,
    scale: float = 0.25,
    **kwargs: Any
) -> torch.nn.Module:
    """
    Creates a QHFT model. Beware print_trainable_parameters() does not work.

    Args:
        model_id(`str`): The model ID for CFQ to be applied
        bits(`int`): quantization bit width. defaults to `4`
        peft_config(`QHFTConfig | None`): configuration for QHFT
        dataset_id(`str`, `optional`): name of the dataset that is used for calibration. defaults to `wikitext2`.
        quant_method(`str`, `optional`): quantization method. defaults to `GPTQ`.
        lora_rank(`int`, `optional`): rank of the LoRA adapter. defaults to `64`.
        group_size(`int`, `optional`): group size for group-wise quantization. defaults to `64`.
        scale(`float`, `optional`): scaling factor for the QHFT adapter. defaults to `0.25`.
    """
    # 1. Apply quantization & 2. Apply PEFT
    peft_model = get_quantized_peft_model(
        model_id,
        quant_method=quant_method,
        bits=bits,
        group_size=group_size,
        rank=lora_rank,
        scale=scale,
        dataset_id=dataset_id
    )

    # 3. Calculate quantization error
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cpu")
    base_model.eval()
    with torch.inference_mode():
        compute_quant_error(peft_model, base_model, quant_method=quant_method)

    # 4. Forward pass using calibration dataset
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = base_model.cuda()
    with torch.inference_mode():
        calibration_forward(base_model, tokenizer, dataset_id)

    for base_module_name, base_module in base_model.named_modules():
        if hasattr(base_module, "xtx_root"):
            for peft_module_name, peft_module in peft_model.named_modules():
                # compare base_module_name and peft_module name
                if peft_module_name == "base_model.model." + base_module_name:
                    peft_module.register_buffer("xtx_root", base_module.xtx_root.cpu())

    del base_model
    torch.cuda.empty_cache()

    # 5. Initialize FourierFT parameters
    with torch.inference_mode():
        initialize_adapter(peft_model, bits=bits, lora_rank=lora_rank)

    return peft_model


def compute_quant_error(quantized_model: torch.nn.Module, base_model: torch.nn.Module, quant_method: str) -> None:
    """
    For each QuantLinear layer in the `quantized_model`, create a new attribute `quant_error` which corresponds to the quantization error of `quantized_model`.
    quant_error is in [in_features, out_features] dimension
    """
    cnt = 0
    quantized_model = quantized_model.cuda()
    device = "cuda"
    for quantized_module_name, quantized_module in tqdm(quantized_model.named_modules(), "Computing quantization error"):
        if hasattr(quantized_module, "qweight"):  # QuantLinear layer
            # Find the layer with same name in the base model layer
            for base_module_name, base_module in base_model.named_modules():
                if quantized_module_name in ["base_model.model." + base_module_name, "base_model.model." + base_module_name + ".base_layer"] and hasattr(base_module, "weight"):

                    # These two parameters are not moved to cuda automatically; the following two lines avoid runtime error
                    quantized_module.wf_unsqueeze_zero = quantized_module.wf_unsqueeze_zero.cuda()
                    quantized_module.wf_unsqueeze_neg_one = quantized_module.wf_unsqueeze_neg_one.cuda()
                    quantized_module.quant_error = (base_module.weight.to(device).T - quantized_module.dequantize_weight().to(device)
                                                    ).cpu()  # set new attribute quant_error. indexed by [in_features, out_features]
                    cnt += 1
    print(f"Computed quantization error for {cnt} layers")


def calibration_forward(model: torch.nn.Module, tokenizer: AutoTokenizer, dataset_id: str, batch_size: int = 1):
    """ This function performs forward pass through `model` using a small calibration dataset, specified by `dataset_id`.
    As a result, each FourierFT layer in `model` attains a new attribute `xtx_root`.

    Args:
        model (`torch.nn.Module`): FourierFT model
        tokenizer (`AutoTokenizer`): Tokenizer used for `model`
        dataset_id (`str`): Name of the dataset. Should be one of ['wikitext2', 'c4', 'c4-new', 'ptb', 'ptb-new'].
        batch_size (`int, optional`): Batch size for forward pass. Defaults to 8.

    """
    # Modified from https://huggingface.co/docs/transformers/perplexity
    # parameter name may differ by model type
    max_length = min(model.config.max_position_embeddings, 4096)
    register_hook(model)

    if dataset_id == "wikitext2":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        encodings = tokenizer("\n\n".join(
            dataset["text"]), return_tensors="pt")
    elif dataset_id == "c4":
        # GPTQ uses C4 train split
        dataset = load_dataset("allenai/c4", "en", split="train")
        encodings = tokenizer("\n\n".join(
            dataset["text"]), return_tensors="pt")
    # TODO: implement these datasets
    elif dataset_id == "c4-new":
        raise NotImplementedError(f"Dataset {dataset_id} is not implemented")
    elif dataset_id == "ptb":
        raise NotImplementedError(f"Dataset {dataset_id} is not implemented")
    elif dataset_id == "ptb-new":
        raise NotImplementedError(f"Dataset {dataset_id} is not implemented")
    else:
        raise NotImplementedError(f"Dataset {dataset_id} is not implemented. "
                                  "dataset_id must be in ['wikitext2', 'c4', 'c4-new', 'ptb', 'ptb-new'].")

    model = model.cuda()
    seq_len = encodings.input_ids.size(1)
    for begin_loc in tqdm(range(0, seq_len, max_length * batch_size),  "Forward passing calibration dataset.."):
        end_loc = min(begin_loc + max_length * batch_size, seq_len)
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        if input_ids.shape[1] < max_length * batch_size:    # truncated
            num_append = max_length * batch_size - input_ids.shape[1]
            input_ids = torch.cat((input_ids, torch.IntTensor(
                [[0] * num_append]).to(input_ids.device)), dim=1)
            target_ids = torch.cat((target_ids, torch.IntTensor(
                [[-100] * num_append]).to(input_ids.device)), dim=1)
        input_ids = input_ids.reshape(batch_size, max_length)
        target_ids = target_ids.reshape(batch_size, max_length)

        with torch.inference_mode():
            model(input_ids, labels=target_ids)

    clear_hook_and_calculate_root(model)
    torch.cuda.empty_cache()


def register_hook(model: torch.nn.Module) -> None:
    """ Install `accumulate_xtx_hook` to FourierFT layers so that they can accumulate X^TX values.
    The hook is called just before the forward call.
    The accumulated values are required to initialize FourierFT parameters in a data-aware manner.
    Args:
        model(`torch.nn.Module`): target module
    """
    def generate_hook(layer_name: str):
        def accumulate_xtx_hook(layer, input: List[torch.Tensor]):
            """
            Accumulate x^T x to layer.xtx_buffer, where x is the input of the layer
            """
            # TODO: the same input is fed to a few layers. e.g., q/k/v projections. exploit the fact to expedite this process.
            # Maybe layer name can be used to discern that the layers share the same input

            # input[0]: shape [Batch size, sequence length, embedding dimension]
            M = input[0].shape[2]
            device = layer.weight.device
            reshaped_input = input[0].reshape(-1, M).to(device)
            if not hasattr(layer, "xtx_buffer"):
                layer.register_buffer("xtx_buffer", torch.zeros(
                    (M, M), dtype=torch.float64))
            xtx = (reshaped_input.T @ reshaped_input).cpu()
            if not xtx.isnan().any() and not xtx.isinf().any():     # prevent NaN or Inf being accumulated to xtx_buffer
                layer.xtx_buffer += xtx
            else:   # retry with float64 multiplication
                reshaped_input = reshaped_input.to(torch.float64)
                xtx = (reshaped_input.T @ reshaped_input).cpu()
                layer.xtx_buffer += xtx

        return accumulate_xtx_hook

    model.hook_handles = []
    for name, module in model.named_modules():
        if name[-4:] == "proj":
            handle = module.register_forward_pre_hook(generate_hook(name))
            model.hook_handles.append(handle)


def clear_hook_and_calculate_root(model: torch.nn.Module) -> None:
    """ This function removes the forward pre hooks installed to capture layer inputs,
    and calculates the matrix root of accumulated X^TX matrices stored in `layer.xtx_buffer`.
    The calculated matrix root is stored in each layer's `layer.xtx_root`.

    Args:
        model (torch.nn.Module): Pytorch model where `register_hook()` is already applied in prior
    """
    num_qhft_layers = len(model.hook_handles)
    for handle in model.hook_handles:
        handle.remove()     # uninstall forward pre-hooks
    del model.hook_handles

    pbar = tqdm(total=num_qhft_layers, desc="Calculating matrix root")
    for _, module in model.named_modules():
        if hasattr(module, "xtx_buffer"):
            pbar.update(1)
            H = module.xtx_buffer.to(model.device)
            # Add 1% of average diagonal values to the diagonals in order to avoid numerical instability
            avg_diag_H = H[range(H.shape[0]), range(H.shape[0])].mean()
            H += torch.diag(torch.Tensor([0.0001 * avg_diag_H] * H.shape[0])).to(H.device)  # 0.01 to 0.0001 for mistrals
            try:
                eigenval, eigenvec = torch.linalg.eigh(H)
            except:
                print(f"SVD Failed; Nan detected?{H.isnan().any()} Inf detected={H.isinf().any()} H=0?{(H==0).all()}")
            # avoid numerical issues producing eigenvalues < 0
            sqrt_eigenval = torch.sqrt(torch.clamp(eigenval, min=0))
            R = torch.diag(sqrt_eigenval) @ eigenvec.T
            module.register_buffer("xtx_root", R.cpu())
            del module.xtx_buffer


def initialize_adapter(model: torch.nn.Module, bits: int = 2,
						match_lora_params: bool = True, lora_rank: int = 64):
    """Initialize qhft spectrum location/values to compensate quantization error

    Args:
        model (torch.nn.Module): PyTorch model with xtx_root and quant_error tagged
        bits (int): quantization bit width. defaults to `2`
        match_lora_params (bool): If True, the number of QHFT adapter parameters in each layer will match that of a LoRA adapter with the specified lora_rank.
        lora_rank (int): When specified, the QHFT adapter will use the same number of parameters as a LoRA adapter with this rank.

    """

    def _allocate_spectrum_to_channels(per_chan_num_spectrum_ratio: torch.Tensor, n_frequency: int, max_spectrum_per_chan: int) -> torch.Tensor:
        num_channels = per_chan_num_spectrum_ratio.shape[0]
        per_chan_num_spectrum_ratio /= per_chan_num_spectrum_ratio.sum()

        num_spectrum_per_chan = (per_chan_num_spectrum_ratio * n_frequency).floor().to(torch.int)

        # channels that exceeds the max number of spectrums per channel
        exceeding_chans = (num_spectrum_per_chan > max_spectrum_per_chan)
        num_exceeding_spectra = (num_spectrum_per_chan[exceeding_chans] - max_spectrum_per_chan).sum()
        num_spectrum_per_chan[exceeding_chans] = max_spectrum_per_chan

        # distribute the remainder to the rest of the channels
        num_nonexceeding_channels = (~exceeding_chans).sum()
        remainder = n_frequency - num_spectrum_per_chan.sum().item()

        num_spectrum_per_chan[~exceeding_chans] += (remainder // num_nonexceeding_channels)
        remainder %= num_nonexceeding_channels

        _, idxs = torch.topk(num_spectrum_per_chan, k=remainder, largest=False)
        num_spectrum_per_chan[idxs] += 1

        return num_spectrum_per_chan

    with torch.no_grad():
        for i, (name, module) in tqdm(enumerate(model.named_modules())):
            # QHFT-adapted layer
            if hasattr(module, "qhft_spectrum"):
                scaling = module.qhft_scaling["default"]
                if match_lora_params:
                    n_frequency = lora_rank * (module.in_features + module.out_features)
                else:
                    n_frequency = module.qhft_n_frequency["default"]

                assert hasattr(module, "base_layer") and hasattr(module, "xtx_root")
                R = module.xtx_root.cuda().to(torch.float32)
                W = module.base_layer.dequantize_weight().to(torch.float32)  # [in_features, out_features]
                assert W.shape[0] == module.in_features and W.shape[1] == module.out_features

                bases = wht(R) / R.shape[-1]

                # Need to represent each column of r_delta_w using linear combination of b columns of N
                # r_delta_w = bases @ X, where each column of X contains a predefine number of nonzeros
                delta_w = module.base_layer.quant_error.cuda()
                r_delta_w = R @ delta_w  # [in, out]

                spectrum_temp = torch.linalg.solve(bases, r_delta_w).cpu()
                assert spectrum_temp.shape == (module.in_features, module.out_features)

                # select n_freq/out_chans top magnitudes from X_temp (considering the magnitude of R Delta W)
                spectrum_indices = []  # list of tuples representing coordinates in (out_chan,in_chan)
                spectrum_values = []

                per_chan_num_spectrum_ratio = (spectrum_temp**2).sum(dim=0).sqrt()**ALPHA
                num_spectrum_per_chan = _allocate_spectrum_to_channels(per_chan_num_spectrum_ratio, n_frequency, module.in_features)

                for i in range(module.out_features):
                    col_vectors_magnitude = ((bases**2).sum(dim=0).cpu() * spectrum_temp[:, i]**2).cpu()
                    _, top_col_indices = torch.topk(
                        col_vectors_magnitude, k=num_spectrum_per_chan[i], dim=0)
                    top_col_indices = top_col_indices.tolist()

                    selected_cols = bases[:, top_col_indices]
                    col_values = torch.linalg.solve(selected_cols.T @ selected_cols, selected_cols.T @ r_delta_w[:, i]).detach().cpu().numpy()

                    spectrum_indices.extend([(i, j) for j in top_col_indices])
                    spectrum_values.extend(col_values.tolist())

                indices = torch.IntTensor(spectrum_indices).transpose(0, 1).contiguous()

                module.qhft_spectrum["default"] = torch.nn.Parameter(
                    torch.Tensor(spectrum_values).cpu(), requires_grad=True
                ) / scaling * sqrt(module.out_features)  # multiply sqrt of out_features to compensate layer.py dividing it again

                module.update_indices("default", indices)

                del module.xtx_root
                del module.base_layer.quant_error
                del r_delta_w
                del bases
                module.xtx_root = None
                module.base_layer.quant_error = None
                r_delta_w = None
                bases = None

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

def find_layer(root: nn.Module, layer_name: str) -> Optional[nn.Module]:
    for name, module in root.named_modules():
        if name == layer_name:
            return module
    return None



if __name__ == "__main__":

    parser = ArgumentParser("Initialize 1D channel-wise DCT or LoRA adapter")
    parser.add_argument("-m", "--model_id", default="meta-llama/Llama-2-7b-hf", type=str, help="target model ID")
    parser.add_argument("-q", "--quant_method", default="gptq", type=str, help="quantization method to use. rtn or gptq")
    parser.add_argument("-b", "--bits", default=4, type=int, help="bit width")
    parser.add_argument("-r", "--rank", default=64, type=int, help="LoRA rank or rank for QHFT to match its number of parameters")
    parser.add_argument("-g", "--group_size", default=64, type=int, help="Group size for group-wise quantization")
    parser.add_argument("-s", "--scale", default=0.25, type=float, help="Scaling factor for adapters")
    parser.add_argument("-e", "--eval_ppl", default=False, type=bool, help="Evaluate perplexity after initialization", action=BooleanOptionalAction)

    # TODO: remove this. just for hyperparameter search
    parser.add_argument("-a", "--alpha", type=float, help="Hyperparameter that affects the spectrum allocation", default=1.0)

    args = parser.parse_args()

    peft_method_name = 'qhft'
    print(f'Initialize {peft_method_name} adapter on {args.model_id} for {args.bits}-bit {args.quant_method}. rank={args.rank}')

    # TODO: remove this
    ALPHA = args.alpha

    QHFT_CACHE_PATH = os.getenv('QHFT_CACHE_PATH', './')
    path_prefix = os.path.join(QHFT_CACHE_PATH, 'initialized_checkpoints')

    if not os.path.exists(path_prefix):
        os.makedirs(path_prefix, exist_ok=True)
    model_name = args.model_id.split('/')[-1]

    path_full = f"{path_prefix}/{args.model_id}-{args.bits}bit-{args.quant_method}-{peft_method_name}-rank{args.rank}{f'-g{args.group_size}' if args.group_size != 64 else ''}"
    print(f"Initialization start. will be saved in {path_full}")

    model = get_initialized_model(model_id=args.model_id, bits=args.bits, dataset_id="wikitext2",
                                  quant_method=args.quant_method,
                                  lora_rank=args.rank, group_size=args.group_size,
                                  scale=args.scale)

    model.save_pretrained(path_full)

    if args.eval_ppl:
        model.cuda()
        for quantized_module_name, quantized_module in model.named_modules():
            if hasattr(quantized_module, "qweight"):  # QuantLinear layer
                # These two parameters are not moved to cuda automatically; the following two lines avoid runtime error
                quantized_module.wf_unsqueeze_zero = quantized_module.wf_unsqueeze_zero.cuda()
                quantized_module.wf_unsqueeze_neg_one = quantized_module.wf_unsqueeze_neg_one.cuda()

        model.seqlen = 2048
        from eval.perplexity_test import eval_ppl
        with torch.no_grad():
            ppl = eval_ppl(args, model, AutoTokenizer.from_pretrained(args.model_id), model.device)
            print(f'{peft_method_name} adapter on {args.model_id} for {args.bits}-bit {args.quant_method}. rank={args.rank}, ALPHA={ALPHA} ppl={ppl}')
