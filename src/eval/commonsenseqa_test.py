import os
import re
import sys
import copy
import json
import argparse
from tqdm import tqdm

import fire

import torch

from transformers import GenerationConfig

from lm_eval import evaluator
from lm_eval.utils import make_table

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def eval_csqa(args, model, tokenizer):

    ### lm-eval-harness version
    task_names = ['hellaswag', 'piqa', 'arc_easy','arc_challenge', 'winogrande', 'boolq', 'openbookqa']
    with torch.no_grad():
        results = evaluator.simple_evaluate(
            model="hf",
            model_args={"pretrained" : model, "tokenizer" : tokenizer},
            tasks=task_names,
            num_fewshot=0,
            batch_size="auto",
            device="cuda",
        )

    return make_table(results)

    ### LLM-adapter version, deprecated
    args.tasks = ["boolq", "piqa", "hellaswag", "winogrande", "ARC-Challenge", "ARC-Easy", "openbookqa"]
    args.batch_size = 48

    def evaluate(
            instructions,
            input=None,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=32,
            **kwargs,
    ):
        prompts = [generate_prompt(instruction, input) for instruction in instructions]
        input_ids = tokenizer(prompts, return_tensors="pt", padding=True)["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences
        outputs = tokenizer.batch_decode(s, skip_special_tokens=True)
        outputs = [o.split("### Response:")[1].strip() for o in outputs]
        return outputs

    print('---------------')
    for task in tqdm(args.tasks, total=len(args.tasks)):
        dataset = load_data(task)
        batches = create_batch(dataset, args.batch_size)

        create_dir('output/')
        save_file = f'output/csqa_eval_result.json'

        total = len(batches)
        correct = 0
        current = 0
        output_data = []
        for idx, batch in tqdm(enumerate(batches), total=len(batches)):
            try:
                current += len(batch)
                instructions = [data.get('instruction') for data in batch]

                outputs = evaluate(instructions)

                for data, output in zip(batch, outputs):
                    label = data.get('answer')
                    flag = False
                    predict = extract_answer(task, output)
                    if label == predict:
                        correct += 1
                        flag = True
                    new_data = copy.deepcopy(data)
                    new_data['output_pred'] = output
                    new_data['pred'] = predict
                    new_data['flag'] = flag
                    output_data.append(new_data)
            except:
                print(f"error raise, passing batch {idx}\n")

        print(f'task: {task} | accuracy: {correct / current * 100:.2f}\n')
    print('---------------')


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""  # noqa: E501


def load_data(task) -> list:
    file_path = f'../eval/dataset/{task}/test.json'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    json_data = json.load(open(file_path, 'r'))
    return json_data

def create_batch(dataset, batch_size):
    batches = []
    num_batch = len(dataset)//batch_size if len(dataset) % batch_size == 0 else len(dataset)//batch_size + 1
    for i in range(num_batch):
        batch = dataset[i*batch_size: min((i+1)*batch_size, len(dataset))]
        batches.append(batch)
    return batches


def extract_answer(task, sentence: str) -> float:
    if task == 'boolq':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'true|false', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif task == 'piqa':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'solution1|solution2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif task in ['social_i_qa', 'ARC-Challenge', 'ARC-Easy', 'openbookqa']:
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'answer1|answer2|answer3|answer4|answer5', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif task == 'hellaswag':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'ending1|ending2|ending3|ending4', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif task == 'winogrande':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'option1|option2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]

