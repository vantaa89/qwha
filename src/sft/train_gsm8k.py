# Modified from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py

import copy
import logging
import os, sys
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Optional, Union

import torch
import transformers
from torch.utils.data import Dataset, random_split
from transformers import Trainer, GPTQConfig, AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers import TrainerCallback, TrainerState, TrainerControl

from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from safetensors.torch import load, load_file

import sft_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import get_quantized_peft_model, load_from_checkpoint
from eval.perplexity_test import eval_ppl
import math
from tqdm import tqdm
import re


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

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
    eval_batch_size: int = field(
        default=256
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    bf16: bool = field(default=True)
    optim: str = field(default="adamw_torch")
    cache_dir: Optional[str] = field(default='/SHARE_ST/vlsi/hf_cache')
    run_name: Optional[str] = field(default=None)
    model_max_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"})
    bits: int = field(default=4)
    group_size: int = field(default=128, metadata={"help": "Quantization group size"})
    peft_init: bool = field(default=True)
    dropout: float = field(default=0.1)
    lora_r: int = field(default=64, metadata={"help": "LoRA rank"})
    scale: float = field(default=3000.0, metadata={"help": "Adapter scale"})
    eval_every_epoch: bool = field(default=False, metadata={"help": "Evaluate accuracy every epoch"})


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


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


def preprocess(sources: Sequence[str], targets: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Preprocess the data by tokenizing."""
    # sources are questions, and targets are answers
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=labels)

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        logging.warning("Formatting inputs...")
        sources = [f"{example['question']}{QUESTION_PROMPT}" for example in raw_data]
        targets = [f"{example['answer']}{tokenizer.eos_token}".replace("####", ANSWER_PROMPT) for example in raw_data]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

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
    logging.warning("Downloading Data")
    dataset = load_dataset(data_args.data_path, "main")
    train_set = dataset['train']
    eval_set = dataset['test']
    train_dataset = SupervisedDataset(raw_data=train_set, tokenizer=tokenizer)
    eval_dataset = SupervisedDataset(raw_data=eval_set, tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


### For evaluation

def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    segment = sentence.split(ANSWER_PROMPT)
    if len(segment) > 1:
        pred_answer = segment[1]
        pred_answer = [s for s in re.findall(r'-?\d+\.?\d*', pred_answer)]
        if len(pred_answer) > 0:
            pred_answer = pred_answer[0]
        else:
            pred_answer = float(pred[-1])
    else:
        # use the last number as the answer
        pred_answer = float(pred[-1])

    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer


def compute_accuracy(pred: list, gold: list):
    acc = 0.0
    for p, g in zip(pred, gold):
        if p == g:
            acc += 1
    return acc / len(pred)


class GenerationEvalCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_batch_size=48, data_path="gsm8k", num_samples=-1):
        self.gen_kwargs = {
            "max_new_tokens": 256,
            "temperature": 0.1,
            "top_k": 40,
            "top_p": 0.95,
            "do_sample": True,
        }

        logging.warning("Downloading Data")
        dataset = load_dataset(data_path, "main")
        test_set = dataset['test']

        self.num_samples = len(test_set) if num_samples == -1 else num_samples
        print(f"Using subset evaluation with {self.num_samples} samples")
        self.tokenizer = tokenizer
        self.eval_batch_size = eval_batch_size

        if self.num_samples < len(test_set):
            test_set = test_set.shuffle(seed=42).select(range(self.num_samples))  # Use a subset of test set to save time

        logging.warning("Formatting inputs...")
        self.question = [f"{example['question']}{QUESTION_PROMPT}" for example in test_set]
        self.answer = []

        # get numerical answer
        for example in test_set['answer']:
            ans = example.split('####')[-1]
            ans = ans.replace(',', '')  # handle numbers like 2,000
            try:
                ans = float(ans)
            except ValueError:
                ans = float("inf")
            self.answer.append(ans)

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs['model']
        metrics = kwargs['metrics']
        model.eval()
        question = self.question
        logging.warning("Tokenizing inputs...")
        eval_step = math.ceil(len(question)/self.eval_batch_size)
        question_data = []

        for i in range(eval_step):
            if i < eval_step - 1:
                batch = self.tokenizer(
                    question[i*self.eval_batch_size: (i+1)*self.eval_batch_size],
                    return_tensors="pt",
                    padding="longest",
                )
            else:
                batch = self.tokenizer(
                    question[i*self.eval_batch_size:],
                    return_tensors="pt",
                    padding="longest",
                )
            batch['input_len'] = len(batch['input_ids'][0])
            question_data.append(batch)


        ans_pred_list = []
        set_seed(42)

        for step, batch in tqdm(enumerate(question_data), total=len(question_data)):
            with torch.no_grad():
                self.gen_kwargs["input_ids"] = batch["input_ids"].to('cuda')
                self.gen_kwargs["attention_mask"] = batch["attention_mask"].to('cuda')
                generated_tokens = model.generate(**self.gen_kwargs)

            pred_tokens = generated_tokens[:, batch['input_len']:]
            decoded_pred = self.tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)

            # Extract the numbers in sentences
            ans_pred_list += [extract_answer_number(sentence_pred) for sentence_pred in decoded_pred]


        accuracy = compute_accuracy(self.answer, ans_pred_list)
        if metrics is not None:
            metrics["eval_subset_acc"] = accuracy
        print(f"Epoch {state.epoch:.6f} eval_subset_acc {accuracy:.6f}")
        model.train()
        return control

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    bits = training_args.bits
    group_size = training_args.group_size
    peft_init = training_args.peft_init

    model_name = model_args.model_name_or_path.split('/')[-1]
    # load smart tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    # load quant init model
    model = get_quantized_peft_model(
        model_args.model_name_or_path, bits=bits,
        group_size=group_size, rank=training_args.lora_r, scale=training_args.scale,
        dropout=training_args.dropout, bf16=training_args.bf16)


    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    # load initialized adapter weights
    if peft_init:
        QHFT_CACHE_PATH = os.getenv('QHFT_CACHE_PATH', './')
        path_prefix = os.path.join(QHFT_CACHE_PATH, f"initialized_checkpoints")
        path_full = f"{path_prefix}/{model_args.model_name_or_path}-{str(bits)}bit-gptq-qhft-rank{training_args.lora_r}"
        load_from_checkpoint(model, path_full, peft_method="qhft", scale=training_args.scale)
        print(f"Load initialized adapter from {path_full}")
        model.print_trainable_parameters()

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    model.eval()
    model.seqlen = 2048
    with torch.no_grad():
        ppl = eval_ppl(None, model, tokenizer, model.device)
        print(f"Perplexity: {ppl}")
    model.train()

    callbacks = []
    if training_args.eval_every_epoch:
        callbacks.append(GenerationEvalCallback(tokenizer, eval_batch_size=data_args.eval_batch_size))

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=callbacks,
        **data_module
    )

    trainer.train()

    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()

