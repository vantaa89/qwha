# Code from https://github.com/NVlabs/MaskLLM/blob/main/eval_llama_ppl.py

# Code adapted from https://github.com/locuslab/wanda/blob/main/main.py
import argparse
from importlib.metadata import version
import os
import time
import fnmatch
import random
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import re
from peft import PeftModelForCausalLM
from utils import get_quantized_peft_model, load_from_checkpoint
from tqdm import tqdm

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    traindata = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    valdata = load_dataset('allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc

# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=4096, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)

# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_ppl(args, model, tokenizer, device=torch.device("cuda:0")):
    # Set dataset
    dataset = "wikitext2"

    # Print status
    print(f"evaluating on {dataset}")

    # Get the test loader
    _, testloader = get_loaders(
        dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer,
    )

    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        ppl_test = eval_ppl_wikitext(model, testloader, 1, device)
    return ppl_test

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext_train(model, trainloader, bs=1, device=None):
    # Get input IDs
    # testenc = testenc.input_ids

    # Calculate number of samples
    # nsamples = testenc.numel() // model.seqlen
    nsamples = len(trainloader)

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in tqdm(range(0,nsamples,bs)):

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        # inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = trainloader[i][0].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext(model, testenc, bs=1, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in tqdm(range(0,nsamples,bs)):

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model_name, cache_dir="./assets/cache"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        device_map="auto"
    )
    model.seqlen = 4096 if model.config.max_position_embeddings>=4096 else model.config.max_position_embeddings
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--mask', type=str, default=None, help="Path to the mask ckpt")
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt",
                        "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", "ablate_wanda_iter", "search"])
    parser.add_argument("--cache_dir", default="/data/hf_cache", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--quant_peft', type=bool, default=False, help='Whether the model is quantized PEFT(LoRA/FourierFT) model')
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    if not args.quant_peft:
        model = get_llm(args.model, args.cache_dir)

        # TODO: to test quantized model
        # from utils import rtn_quantize_model
        # model.cpu()
        # rtn_quantize_model(model, 3)
        # for quantized_module_name, quantized_module in model.named_modules():
        #     if hasattr(quantized_module, "qweight"):  # QuantLinear layer
        #         # These two parameters are not moved to cuda automatically; the following two lines avoid runtime error
        #         quantized_module.wf_unsqueeze_zero = quantized_module.wf_unsqueeze_zero.cuda()
        #         quantized_module.wf_unsqueeze_neg_one= quantized_module.wf_unsqueeze_neg_one.cuda()
        # model.cuda()

        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    else:
        model_id, bits, quant_method, peft_method, rank = parse_checkpoint(args.model.split('/')[-1])
        if "llama" in model_id.lower():
            model_id = '/'.join(["meta-llama", model_id])
        if peft_method.lower() != "lora":
            model = get_quantized_peft_model(model_id, quant_method=quant_method, bits=bits, peft_method=peft_method, rank=rank)
            load_from_checkpoint(model, os.path.join(args.model, 'adapter_model.safetensors'), peft_method=peft_method)
        else:   # for LoRA, load_for_checkpoint is not working. load it manually
            model = AutoModelForCausalLM.from_pretrained(args.model, device_map="cpu") # This attaches trained lora weight on top of unquantized weight
            from utils import rtn_quantize_model
            rtn_quantize_model(model)
            model.cuda()
            # import pdb; pdb.set_trace()
            for quantized_module_name, quantized_module in model.named_modules():
                if hasattr(quantized_module, "qweight"):  # QuantLinear layer
                    # These two parameters are not moved to cuda automatically; the following two lines avoid runtime error
                    quantized_module.wf_unsqueeze_zero = quantized_module.wf_unsqueeze_zero.cuda()
                    quantized_module.wf_unsqueeze_neg_one= quantized_module.wf_unsqueeze_neg_one.cuda()
        model.seqlen = 4096 if model.config.max_position_embeddings>=4096 else model.config.max_position_embeddings
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

    if args.mask is not None:
        if args.mask.endswith(".pt"): # raw mask ckpt, this will be quite large (~6GB for 7b model)
            mask_ckpt = torch.load(args.mask, map_location='cpu')
            model_state = model.state_dict()
            for k, v in mask_ckpt.items():
                k_original = k.replace(".mask", "")
                model_state[k_original] *= v.to(model_state[k_original].device).float()
            model.load_state_dict(model_state)
        else:
            if args.mask.endswith(".npz"): # compressed mask ckpt, this will be much smaller (~500MB for 7b model)
                mask_ckpt = np.load(args.mask)
            else:
                from huggingface_hub import hf_hub_download
                downloaded_mask = hf_hub_download(repo_id=args.mask, filename="mask_compressed.npz")
                mask_ckpt = np.load(downloaded_mask)
            model_state = model.state_dict()
            for k, v in mask_ckpt.items():
                k_original = k.replace(".mask", "")
                v = np.unpackbits(v) # to bits
                mask = torch.from_numpy(v).to(model_state[k_original].device).float()
                mask = mask.view(*model_state[k_original].shape) # reshape the mask
                model_state[k_original] *= mask # apply the mask
            model.load_state_dict(model_state)
    model.eval()

    for name, param in model.named_parameters():
        sparsity = (param==0).float().mean().item()
        print(f"{name} - sparsity {sparsity:.4f}")
        # Check 2:4
        if abs(sparsity-0.5)<0.0001:
            param_reshaped = param.reshape(-1, 4)
            mask_sum = (param_reshaped==0).sum(dim=-1)
            assert (mask_sum>=2).all()

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)
    ppl_test = eval_ppl(args, model, tokenizer, device)
    print(f"wikitext perplexity for rank={rank} {peft_method} {bits}b {quant_method}: {ppl_test}")

    if args.save_model is not None:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)


def parse_checkpoint(s):
    if '/' in s:
        s = s.split('/')[-1]
    pattern = r'(?P<model_id>.+?)-(?P<number>\d+)bit-(?P<quant>\w+)-(?P<peft>\w+)-rank(?P<rank>\d+)'
    match = re.match(pattern, s)

    if match:
        model_id = match.group("model_id")
        number = int(match.group('number'))
        quant = match.group('quant')
        peft = match.group('peft')
        rank = int(match.group('rank'))
        return model_id, number, quant, peft, rank

    return None


if __name__ == '__main__':
    main()
