#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
import os, sys
import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Optional, Union

import torch
import transformers
from torch.utils.data import Dataset, random_split
from transformers import Trainer, GPTQConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, FourierFTConfig, TaskType, get_peft_model, LoraConfig

from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from safetensors.torch import load, load_file
from transformers import set_seed

import sft_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import get_quantized_peft_model, load_from_checkpoint
import math
from tqdm import tqdm

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

FEW_SHOT_PROMPT = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nAnswer the above question. First think step by step and then answer the final number.\n
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The final answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nAnswer the above question. First think step by step and then answer the final number.\n
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The final answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nAnswer the above question. First think step by step and then answer the final number.\n
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The final answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nAnswer the above question. First think step by step and then answer the final number.\n
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The final answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nAnswer the above question. First think step by step and then answer the final number.\n
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The final answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nAnswer the above question. First think step by step and then answer the final number.\n
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The final answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nAnswer the above question. First think step by step and then answer the final number.\n
A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The final answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nAnswer the above question. First think step by step and then answer the final number.\n
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The final answer is 8.

Q: {question}\nAnswer the above question. First think step by step and then answer the final number.\n
A:"""
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
    ckpt_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to your local output directory"},
    )
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to your local output directory"},
    )
    bits: int = field(default=4)
    group_size: int = field(default=128, metadata={"help": "Quantization group size"})
    quant_method: str = field(default="gptq", metadata={"help": "Quantization method"})
    lora_r: int = field(default=64, metadata={"help": "LoRA rank"})
    scale: float = field(default=1.0, metadata={"help": "Adapter scale"})




@dataclass
class DataArguments:
    data_path: str = field(
        default="gsm8k",
        metadata={"help": "Dataset path."}
    )

    batch_size: int = field(
        default=32,
        metadata={"help": "Evaluation batch size."}
    )
    model_max_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default='/SHARE_ST/vlsi/hf_cache')
    run_name: Optional[str] = field(default=None)


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


def evaluation(model_args, data_args):
    bits = model_args.bits
    quant_method = model_args.quant_method
    group_size = model_args.group_size

    # load smart tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=data_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    # load quant init model
    model = get_quantized_peft_model(model_args.model_name_or_path, bits=bits, quant_method=quant_method, group_size=group_size,
                rank=model_args.lora_r, bf16=False)

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

    if model_args.adapter_name_or_path is not None:
        load_from_checkpoint(model, model_args.adapter_name_or_path, scale=model_args.scale)


    ######################
    #      dataset       #
    ######################
    logging.warning("Downloading Data")
    dataset = load_dataset(data_args.data_path, "main")
    test_set = dataset['test']

    logging.warning("Formatting inputs...")
    question = [f"{example['question']}{QUESTION_PROMPT}" for example in test_set]
    # question = [FEW_SHOT_PROMPT.format(question=example['question']) for example in test_set]
    answer = []

    # get numerical answer
    for example in test_set['answer']:
        ans = example.split('####')[-1]
        ans = ans.replace(',', '')  # handle numbers like 2,000
        try:
            ans = float(ans)
        except ValueError:
            ans = float("inf")
        answer.append(ans)

    logging.warning("Tokenizing inputs...")
    eval_step = math.ceil(len(question)/data_args.batch_size)
    logging.warning(f"Total example: {len(question)} | eval batch size: {data_args.batch_size} | "
                    f"eval steps: {eval_step}")
    question_data = []
    for i in range(eval_step):
        if i < eval_step - 1:
            batch = tokenizer(
                question[i*data_args.batch_size: (i+1)*data_args.batch_size],
                return_tensors="pt",
                padding="longest",
            )
        else:
            batch = tokenizer(
                question[i*data_args.batch_size:],
                return_tensors="pt",
                padding="longest",
            )
        batch['input_len'] = len(batch['input_ids'][0])
        question_data.append(batch)

    model.eval()
    gen_kwargs = {
        "max_new_tokens": 256,
        "temperature": 0.1,
        "top_k": 40,
        "top_p": 0.95,
        "do_sample": True,
    }
    ans_pred_list = []
    set_seed(42)
    for step, batch in tqdm(enumerate(question_data), total=len(question_data)):
        with torch.no_grad():
            gen_kwargs["input_ids"] = batch["input_ids"].to('cuda')
            gen_kwargs["attention_mask"] = batch["attention_mask"].to('cuda')
            generated_tokens = model.generate(**gen_kwargs)

        pred_tokens = generated_tokens[:, batch['input_len']:]
        decoded_pred = tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)

        # Extract the numbers in sentences
        # print(decoded_pred)
        ans_pred_list += [extract_answer_number(sentence_pred) for sentence_pred in decoded_pred]
        # print(f"cummulative acc batch {step}\t {compute_accuracy(answer[:len(ans_pred_list)], ans_pred_list):.4f}")

    # print("prediction", ans_pred_list)
    # print("ground truth", answer)

    accuracy = compute_accuracy(answer, ans_pred_list)

    print(f"adapter: {model_args.adapter_name_or_path} | GSM8K test accuracy: {100*accuracy:.2f}% | "
          f"full precision: {model_args.full_precision}")


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


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    if model_args.ckpt_dir is not None:
        adapter_dir_list = [os.path.join(model_args.ckpt_dir, ckpt_dir) for ckpt_dir in os.listdir(model_args.ckpt_dir)
                            if 'checkpoint-' in ckpt_dir]
    elif model_args.adapter_name_or_path is not None:
        adapter_dir_list = [model_args.adapter_name_or_path]
    else:
        logging.warning("Use the checkpoint in HF hub, stored in the `subfolder='gsm8k'` in target model.")
        adapter_dir_list = [None]

    for adapter_path in adapter_dir_list:
        model_args.adapter_name_or_path = adapter_path
        evaluation(model_args, data_args)

