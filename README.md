# QHFT: Quantization-Aware Walsh-Hadamard Adaptation for Parameter-Efficient Fine-Tuning on Large Language Models

This is the official implementation of QHFT (Quantization-Aware Walsh-Hadamard Adaptation for Parameter-Efficient Fine-Tuning).

## Overview

QHFT is a parameter-efficient fine-tuning method designed for quantized large language models. It leverages Walsh-Hadamard transformations to adapt quantized models efficiently while acheiving high fine-tuned accuracy.

## Installation

First, install `uv` if not already installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install dependencies:

```bash
uv sync
source .venv/bin/activate
uv pip install gptqmodel==2.2.0 --no-build-isolation
cd peft
uv pip install -e .
```

## Initialization

Before proceeding with the initialization steps, set the `QHFT_CACHE_PATH` environment variable. This directory will store quantized models and QHFT-initialized checkpoints:

```bash
export QHFT_CACHE_PATH=/path/to/cache
```

### Run MagR Quantization

First, run [MagR](https://github.com/aozhongzhang/magr) quantization. Below is an example for `meta-llama/Llama-3.2-3B`, quantized to 2-bit per-group quantization with group size 64. A shell script is also provided in `MagR/quant.sh`:

```bash
MODEL_ID=meta-llama/Llama-3.2-3B
BITS=2
GROUPS=64
python MagR/llama.py $MODEL_ID --wbits $BITS --groupsize $GROUPS --magr --static-groups --save "${QHFT_CACHE_PATH}/gptq_models/$MODEL_ID-${BITS}bits-g${GROUPS}"
```

### Run QHFT Initialization

Next, run the QHFT initialization code. A shell script is also provided in `src/init/init.sh`:

```bash
MODEL_ID=meta-llama/Llama-3.2-3B
BITS=2
GROUPS=64
RANK=64
python src/init/initialize.py -m $MODEL_ID -q gptq -b $BITS -g $GROUPS -r $RANK --eval_ppl
```

## Fine-Tuning

Use the shell scripts in `src/sft` to fine-tune QHFT models on each datasets:

```bash
cd src/sft/
./train_gsm8k.sh    # Train on GSM8K dataset
./train_alpaca.sh   # Train on Alpaca dataset
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{qhft2025,
  title={QHFT: Quantization-Aware Walsh-Hadamard Adaptation for Parameter-Efficient Fine-Tuning on Large Language Models},
  author={Hyesung Jeon and Seojune Lee and BeomSeok Kang and Yulhwa Kim and Jae-Joon Kim},
  journal={arXiv preprint},
  year={2025}
}
```

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

