# Modified from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py

import copy
import logging
import os, sys
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Optional, Union

import sys

import torch
import transformers
from torch.utils.data import Dataset, random_split
from transformers import Trainer, GPTQConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, FourierFTConfig, TaskType, get_peft_model, LoraConfig

from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from safetensors.torch import load, load_file

import sft_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import get_quantized_peft_model, load_from_checkpoint
from eval import eval_ppl, eval_csqa, eval_math

IGNORE_INDEX = -100
# DEFAULT_PAD_TOKEN = "[PAD]"
# DEFAULT_EOS_TOKEN = "</s>"
# DEFAULT_BOS_TOKEN = "<s>"
# DEFAULT_UNK_TOKEN = "<unk>"

# alpaca
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
# commonsenseqa
COMPLETION_PROMPT = "the correct answer is "
# gsm8k
QUESTION_PROMPT = "\nAnswer the above question. First think step by step and then answer the final number.\n"
ANSWER_PROMPT = "The final answer is: "

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={"help": "Path to the model."},
    )
    full_precision: bool = field(
        default=True,
        metadata={"help": "Apply QLoRA-style 4-bit quantization if False"}
    )

@dataclass
class DataArguments:
    data_path: str = field(
        default=None,
        metadata={"help": "Dataset path."}
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    bf16: bool = field(default=True)
    optim: str = field(default="adamw_torch")
    cache_dir: Optional[str] = field(default='/SHARE_ST/vlsi/hf_cache')
    run_name: Optional[str] = field(default=None)
    load_best_model_at_end: bool = field(default=False)
    model_max_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"})
    bits: int = field(default=4)
    group_size: int = field(default=128, metadata={"help": "Quantization group size"})
    peft_init: bool = field(default=True)
    lora_r: int = field(default=64, metadata={"help": "LoRA rank"})
    scale: float = field(default=3000.0, metadata={"help": "FourierFT scale"})
    # train and evaluation : later move to additional arguments
    finetune: bool = field(default=True)
    eval_csqa: bool = field(default=False)
    eval_ppl: bool = field(default=False)
    eval_math: bool = field(default=False)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
        model.get_input_embeddings().requires_grad_(False)
        model.get_output_embeddings().requires_grad_(False)


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(sources: Sequence[str], targets: Sequence[str], raw: None, tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Preprocess the data by tokenizing."""
    if raw is not None:
        # no question-answering
        examples= [line for line in raw['text'] if (len(line) > 100 and not line.isspace())]
        examples_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples)]
        input_ids = examples_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
    else:
        # sources are questions, and targets are answers
        examples = [s + t for s, t in zip(sources, targets)]
        examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
        input_ids = examples_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        # we consider including source as a training target
        # for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            # label[:source_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path: str, raw_data: None):
        super(SupervisedDataset, self).__init__()

        logging.warning("Formatting inputs...")
        sources, targets, raw = None, None, None
        if data_path is not None and "alpaca" in data_path:
            prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
            sources = [
                prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
                for example in raw_data
            ]
            targets = [f"{example['output']}{tokenizer.eos_token}" for example in raw_data]

        elif "commonsense" in data_path:
            sources = [f"{example['instruction']}" for example in raw_data]
            targets = [f"{example['output']}{tokenizer.eos_token}" for example in raw_data]

        elif "gsm8k" in data_path:
            sources = [f"Q: {example['question']}{QUESTION_PROMPT}" for example in raw_data]
            targets = [f"A: {example['answer']}{tokenizer.eos_token}".replace("####", ANSWER_PROMPT) for example in raw_data]

        elif "wikitext" in data_path:
            raw = raw_data

        else:
            raise NotImplementedError("Dataset out of configuration. Select among alpaca, commonsense, gsm8k, and wikitext")

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, raw, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    logging.warning("Loading data...")
    if "alpaca" in data_args.data_path:
        dataset_raw = sft_utils.jload("alpaca_data_cleaned.json")
        dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, raw_data=dataset_raw)
        train_dataset, eval_dataset = random_split(dataset, [0.8, 0.2])
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    elif "commonsense" in data_args.data_path:
        data_args.data_path = "zwhe99/commonsense_170k"
        dataset_raw = load_dataset(data_args.data_path)["train"]
        dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, raw_data=dataset_raw)
        train_dataset, eval_dataset = random_split(dataset, [0.9, 0.1])
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    elif "gsm8k" in data_args.data_path:
        train_dataset_raw = load_dataset(data_args.data_path, "main")["train"]
        eval_dataset_raw = load_dataset(data_args.data_path, "main")["test"]
        train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, raw_data=train_dataset_raw)
        eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, raw_data=eval_dataset_raw)
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    elif "wikitext" in data_args.data_path:
        train_dataset_raw = load_dataset(data_args.data_path, "wikitext-2-v1")["train"]
        eval_dataset_raw = load_dataset(data_args.data_path, "wikitext-2-v1")["test"]
        train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, raw_data=train_dataset_raw)
        eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, raw_data=eval_dataset_raw)
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, mlm=False)

    else:
        raise NotImplementedError("Dataset out of configuration. Select among alpaca, commonsense, gsm8k, and wikitext")

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)

model_id_dict = {
    "llama-2-7b" : "meta-llama/Llama-2-7b-hf",
    "llama-3.1-8b" : "meta-llama/Llama-3.1-8B",
    "llama-3.2-3b" : "meta-llama/Llama-3.2-3B",
    "mistral-7b-v0.3" : "mistralai/Mistral-7B-v0.3",
}

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    bits = training_args.bits
    group_size = training_args.group_size
    peft_init = training_args.peft_init

    model_name = model_args.model_name_or_path.split('/')[-1]
    original_model_id = model_id_dict[model_name.lower()]
    # load smart tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        original_model_id,
        model_max_length=training_args.model_max_length,
        padding_side="left",
    )

    # load quant init model
    model = get_quantized_peft_model(model_args.model_name_or_path, bits=bits, group_size=group_size, rank=training_args.lora_r, scale=training_args.scale, bf16=training_args.bf16 and training_args.finetune)
    model.config.use_cache = False

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if training_args.finetune:
        # disable gradient checkpointing for compute-efficiency
        model.config.gradient_checkpointing = False

        # load initialized adapter weights
        if peft_init:
            QHFT_CACHE_PATH = os.getenv('QHFT_CACHE_PATH', './')
            path_prefix = os.path.join(QHFT_CACHE_PATH, f"initialized_checkpoints")
            path_full = f"{path_prefix}/{model_args.model_name_or_path}-{str(bits)}bit-gptq-qhft-rank{training_args.lora_r}"
            load_from_checkpoint(model, path_full, peft_method="qhft", scale=training_args.scale)
            print(f"Load initialized adapter from {path_full}")
            model.print_trainable_parameters()

        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
        trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
        # print(trainer.evaluate())
        trainer.train()
        trainer.save_state()
        trainer.save_model(output_dir=training_args.output_dir)

        print(f"{torch.cuda.max_memory_allocated() / (1000 ** 3) :.2f} GB")

    if not training_args.finetune:
        if peft_init:
            path_prefix = "/SHARE_ST/vlsi/fourierft/initialized_checkpoints"
            load_from_checkpoint(model, f"{path_prefix}/{model_name}-{str(bits)}bit-gptq-qhft-rank{training_args.lora_r}", peft_method="qhft", scale=training_args.scale)
            print(f"Load initialized adapter from {path_prefix}/{model_name}-{str(bits)}bit-gptq-qhft-rank{training_args.lora_r}")
        else:
            load_from_checkpoint(model, training_args.output_dir, peft_method="qhft", scale=training_args.scale)
            print(f"Load fine-tuned adapter from checkpoint {training_args.output_dir}")


    model.eval()
    tokenizer.model_max_length = 2048

    if training_args.eval_csqa:
        scores = eval_csqa(training_args, model, tokenizer)
        print(scores)

    if training_args.eval_ppl:
        model.seqlen = 2048
        ppl = eval_ppl(training_args, model, tokenizer)
        print(f"The perplexity of wikitext2 : {ppl}")

    if training_args.eval_math:
        scores = eval_math(training_args, model, tokenizer)
        print(scores)


if __name__ == "__main__":
    train()
