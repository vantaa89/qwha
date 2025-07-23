import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTQConfig
import os
import torch
import torch.nn as nn
from safetensors.torch import load_file, load
import typing
from peft import PeftModel, QHFTConfig, TaskType, get_peft_model, LoraConfig
import warnings
import json
# from gptqmodel import GPTQModel, QuantizeConfig
from datasets import load_dataset
import random

QHFT_CACHE_PATH = os.getenv('QHFT_CACHE_PATH', './')
if not os.path.exists(QHFT_CACHE_PATH):
    os.makedirs(QHFT_CACHE_PATH, exist_ok=True)

GPTQ_CACHE_PATH = os.path.join(QHFT_CACHE_PATH, "gptq_models/")
RTN_CACHE_PATH = os.path.join(QHFT_CACHE_PATH, "rtn_models/")

def get_quantized_peft_model(
        model_id: str,
        bits: int = 4,
        quant_method: str = "gptq",
        group_size: int = 128,
        rank: int = 64,
        scale: float = 3000.0,
        peft_config: QHFTConfig | LoraConfig | None = None,
        dataset_id: str = "wikitext2",
        dropout: float = 0.0,
        bf16: bool = True) -> torch.nn.Module:
    """
    Returns a torch model which applies both PEFT and Quantization(RTN/GPTQ)

    Args:
        model_id (str): model path or ID
        bits (int, optional): Quantization bit width. Defaults to 4.
        peft_method (str, optional): PEFT method. Defaults to "qhft".
        peft_config (QHFTConfig | LoraConfig | None, optional): PeftConfig object.
        dataset_id (str, optional): Defaults to "wikitext2".
        rank (int, optional): LoRA rank. For QHFT, n_frequency is set to match the LoRA with such rank. Defaults to 64.
        quant_method (str, optional): Quantization method. "GPTQ" or "RTN". Defaults to "GPTQ".

    Returns:
        torch.nn.Module: A torch.nn.Module where PEFT and Quantization are both applied
    """

    # 1. setup peft_config

    if peft_config is None:
        warnings.warn(
            "No peft_config given. "
            "Applying default configuration."
        )
        TARGET_MODULES = ["q_proj", "k_proj", "v_proj",
                          "o_proj", "up_proj", "down_proj", "gate_proj"]
        print("QHFT config")
        peft_config = QHFTConfig(
            task_type=TaskType.CAUSAL_LM,
            n_frequency=524288,         # 2 * 64 * 4096. to match number of parameters in CloQ
            target_modules=TARGET_MODULES,
            scaling=scale,
            random_loc_seed=777,
            init_weights=True,          # initialize to zero
        )
    # 2. Determine the order of PEFT and quantization based on the PEFT method
    # apply peft
    if quant_method.lower() == "gptq":
        print("GPTQ Quantize")
        quantized_model_id = f"{model_id}-{bits}bits-g{group_size}"
        if os.path.exists(f"{GPTQ_CACHE_PATH}/{quantized_model_id}"):
            print(f"Found existing model {GPTQ_CACHE_PATH}/{quantized_model_id}")
            quantized_model = AutoModelForCausalLM.from_pretrained(f"{GPTQ_CACHE_PATH}/{quantized_model_id}", device_map="auto")
        else:
            print(f"No pre-quantized model {quantized_model_id}. Start GPTQ.")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            quantized_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=GPTQConfig(
                    bits=bits,
                    sym=False,
                    dataset=dataset_id,
                    tokenizer=tokenizer,
                    group_size=group_size,
                ),
                device_map="cuda"
            )
            quantized_model.save_pretrained(f"{GPTQ_CACHE_PATH}/{quantized_model_id}")

    elif quant_method.lower() == "rtn":
        print("RTN Quantize")
        quantized_model_id = f"{model_id}-{bits}bits-g{group_size}"
        if False:  # os.path.exists(f"{RTN_CACHE_PATH}/{quantized_model_id}"):  # TODO solve rtn save and load unstablility
            print(f"Found existing model {RTN_CACHE_PATH}/{quantized_model_id}")
            quantized_model = AutoModelForCausalLM.from_pretrained(f"{RTN_CACHE_PATH}/{quantized_model_id}", device_map="auto")
        else:
            print(f"No pre-quantized model {quantized_model_id}. Start RTN.")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            quantized_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu")
            quantized_model.quantization_method = "gptq"
            quantized_model.config.quantization_config = GPTQConfig(
                bits=bits,
                sym=False,
                dataset=dataset_id,
                tokenizer=tokenizer,
                group_size=group_size,
            )
            rtn_quantize_model(quantized_model, bits=bits, group_size=group_size)
            quantized_model.save_pretrained(f"{RTN_CACHE_PATH}/{quantized_model_id}")

    else:
        print("Quantization not applied. Load pre-trained model")
        quantized_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda")

    # apply quantization
    quantized_peft_model = get_peft_model(quantized_model, peft_config)

    # Assign model dtype into bfloat for bf16 training that prevents OOM
    if bf16:
        quantized_peft_model.to(torch.bfloat16)

    # Adjust n_frequency layer-wise for qhft
    for name, module in quantized_peft_model.named_modules():
        if hasattr(module, "qhft_spectrum"):
            module.qhft_n_frequency = rank * (module.in_features + module.out_features)
            module.qhft_spectrum['default'] = torch.nn.Parameter(torch.zeros(module.qhft_n_frequency).cuda(), requires_grad=True)
            module.qhft_indices['default'] = torch.randperm(
                module.out_features * module.in_features,
                generator=torch.Generator().manual_seed(module.qhft_random_loc_seed["default"]),
            )[:module.qhft_n_frequency]
            module.qhft_indices["default"] = torch.stack([
                module.qhft_indices["default"] // module.in_features,
                module.qhft_indices["default"] % module.in_features
            ], dim=0)


    for quantized_module_name, quantized_module in quantized_peft_model.named_modules():
        if hasattr(quantized_module, "qweight"):  # QuantLinear layer
            # These two parameters are not moved to cuda automatically; the following two lines avoid runtime error
            quantized_module.wf_unsqueeze_zero = quantized_module.wf_unsqueeze_zero.cuda()
            quantized_module.wf_unsqueeze_neg_one= quantized_module.wf_unsqueeze_neg_one.cuda()

    return quantized_peft_model.cuda()


def test_model(peft_model: torch.nn.Module, tokenizer: AutoTokenizer) -> None:
    prompt = "Washington D.C. is the capital of"

    # prompt를 토크나이즈 후 모델의 디바이스로 이동
    inputs = tokenizer([prompt, prompt], return_tensors="pt").to(
        peft_model.device)

    generate_ids = peft_model.generate(
        inputs.input_ids,
        max_length=60,
        do_sample=True,
        temperature=0.8
    )

    result = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print("Generated text:", result[0])


def rtn_quantize_model(model: torch.nn.Module | torch.Tensor, bits:int=4, group_size:int=128) -> None:

    from gptqmodel.utils.importer import hf_select_quant_linear
    QuantLinear = hf_select_quant_linear(
        desc_act=False,
        group_size=group_size,
        bits=bits,
        sym=False,
        checkpoint_format="safetensors",
        device_map="cuda",
    )

    def _replace_linear(module: nn.Module):
        # Iterate over immediate children of the module.
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and \
                "lm_head" not in name and "lora" not in name:
                # Create new QuantLinear with the same parameters as the original Linear layer.
                quant_linear = QuantLinear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=(child.bias is not None),
                    bits=bits,
                    group_size=group_size,
                    desc_act=False,
                    sym=False
                )
                _, scale, zero = rtn_quantize(child, bits, group_size, return_scale_zero=True)
                quant_linear.pack(child.cpu(), scales=scale, zeros=zero)
                quant_linear.post_init()

                # Replace the Linear module with the new QuantLinear instance.
                setattr(module, name, quant_linear)
            else:
                # Recursively check child modules.
                if "lora" not in name:
                    _replace_linear(child)

    _replace_linear(model)
    for quantized_module_name, quantized_module in model.named_modules():
        if hasattr(quantized_module, "qweight"):  # QuantLinear layer
            # These two parameters are not moved to cuda automatically; the following two lines avoid runtime error
            quantized_module.wf_unsqueeze_zero = quantized_module.wf_unsqueeze_zero.cuda()
            quantized_module.wf_unsqueeze_neg_one= quantized_module.wf_unsqueeze_neg_one.cuda()

    model.to("cuda")
    model.config.quantization_config.tokenizer = None


def rtn_quantize(layer: torch.nn.Module | torch.Tensor, bits:int=4, group_size:int=128, return_scale_zero=False) -> torch.Tensor:
    """
    Quantize a PyTorch nn.Linear layer using round-to-nearest quantization scheme.
    Replaces `layer` into a quantized version and returns the quantized weight tensor.

    Args:
        layer (torch.nn.Module): pytorch module
        bits (int, optional): bit width of quantization. Defaults to 4.
        group_size (int, optional): group size. Defaults to 128.
        inplace (bool, optional): whether to replace the existing weight. Defaults to False

    Returns:
        `torch.Tensor`: quantized weight tensor.
    """
    if not isinstance(layer, nn.Linear) and not isinstance(layer, torch.Tensor):
        raise NotImplementedError(f"Input type {type(layer)} is not yet implemented")
    with torch.no_grad():
        if isinstance(layer, torch.nn.Module):
            weight = layer.weight
        else:
            weight = layer
        out_features, in_features = weight.shape

        num_group = (in_features + group_size - 1) // group_size

        try:
            weight_grouped = weight.reshape(out_features * num_group, group_size)
        except:
            raise ValueError("Quantization group size {group_size} is not a divisor of the in_features {in_features}")
        weight_min = torch.clamp(weight_grouped.min(dim=1, keepdim=True)[0], max=0)
        weight_max = torch.clamp(weight_grouped.max(dim=1, keepdim=True)[0], min=0)

        # followed https://github.com/ModelCloud/GPTQModel/blob/511f83f202abfd4975bbaf2e4823070529bb172a/gptqmodel/quantization/quantizer.py#L79C29-L79C30
        tmp = (weight_min == 0) & (weight_max == 0)
        weight_min[tmp] = -1
        weight_max[tmp] = 1

        # compute scale and zero point
        scale = (weight_max - weight_min) / (2**bits - 1)
        zero_point = torch.round(-weight_min / scale)

        qweight = torch.clamp(torch.round(weight_grouped / scale + zero_point), 0, 2**bits-1)
        fake_quantized_weight = (qweight - zero_point) * scale
        qweight = qweight.view(in_features, out_features).transpose(0,1)
    if return_scale_zero:
        return fake_quantized_weight.reshape(out_features, in_features), scale.reshape(out_features, num_group), zero_point.reshape(out_features, num_group)
    else:
        return fake_quantized_weight.reshape(out_features, in_features)

def load_from_checkpoint(model:torch.nn.Module, path:str, peft_method:str="qhft", scale=None):
    try:
        checkpoint = load_file(os.path.join(path, "adapter_model.safetensors"))
    except:
        with open(os.path.join(path, "adapter_model.safetensors"), "rb") as f:
            content = f.read()
        checkpoint = load(content)

    with open(os.path.join(path, "adapter_config.json"), "r") as f:
        adapter_config = json.load(f)

    if peft_method.lower() == "qhft":
        for name, module in model.named_modules():
            if hasattr(module, "qhft_spectrum"):
                checkpoint_key = f"{name}.qhft_spectrum"
                module.qhft_spectrum["default"].data = checkpoint[checkpoint_key].data.to(model.device)
                checkpoint_key = f"{name}.qhft_indices_default"
                module.qhft_indices["default"].data = checkpoint[checkpoint_key].data.to(model.device)
                if scale is not None:
                    module.qhft_scaling["default"] = scale
                    module.qhft_spectrum["default"].data *= adapter_config['scaling'] / scale
                else:
                    module.qhft_scaling["default"] = adapter_config['scaling']
