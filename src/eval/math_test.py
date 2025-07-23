import re
import os
import json

import math
import random
from tqdm import tqdm

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteriaList,
    StoppingCriteria
)
from datasets import load_dataset
from collections import Counter


# Define a stopping condition for generation
class SpecificStringStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings, input_len):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings
        self.input_len = input_len

    def __call__(self, input_ids, scores, **kwargs):
        text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        current_text = [t[self.input_len:] for t in text]

        return all(any(stop_string in text for stop_string in self.stop_strings) for text in current_text)


def extract_predicted_answer(text):
    regex_pattern = "(-?[$0-9.,]{2,})|(-?[0-9]+)"
    regexes_to_ignore =[
        ",",
        "\\$",
        "(?s).*#### ",
        "\\.$"
    ]
    match = re.findall(regex_pattern, text)
    if match:
        match = match[-1]
        if isinstance(match, tuple):
            match = [m for m in match if m][0]
        text = match.strip()

        for regex in regexes_to_ignore:
            text = re.sub(regex, "", text)
        try:
            result = float(text.replace(',', '').replace("..", ''))
        except:
            result = float('-inf')
        return result
    else:
        return None

def extract_ground_truth(text):
    return float(text.split('####')[-1].strip().replace(',', ''))


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


def evaluation(args, model, tokenizer):
    random_seed = 42
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    if args.quant_method in ["rtn", "gptq"]:
        batch_size = 24
    else:
        batch_size = 16

    print('\nLoading dataset...')
    dataset = load_dataset('gsm8k', "main", split='test')
    datasize = len(dataset)
    print('gsm8k test size:', datasize)

    # Define a stopping condition for generation
    generation_util = [
        "Q:",
        "</s>",
        "<|im_end|>"
    ]
    eval_step = math.ceil(datasize / batch_size)

    input_text, inputs, ground_truth_answer = [], [], []
    for i in tqdm(range(datasize), desc='Data formatting'):
        example = dataset[i]
        input_text.append(FEW_SHOT_PROMPT.format(question=example['question']))
        ground_truth_answer.append(extract_ground_truth(example['answer']))

    text_len = len(FEW_SHOT_PROMPT.format(question=""))
    stop_criteria = SpecificStringStoppingCriteria(tokenizer, generation_util, text_len)
    stopping_criteria_list = StoppingCriteriaList([stop_criteria])

    inputs = []
    for i in tqdm(range(eval_step), desc='Data batching'):
        if i < eval_step - 1:
            batch = tokenizer(
                input_text[i*batch_size: (i+1)*batch_size],
                return_tensors="pt",
                padding="longest",
            )
        else:
            batch = tokenizer(
                input_text[i*batch_size:],
                return_tensors="pt",
                padding="longest",
            )
        batch['bsz'], batch['input_len'] = batch['input_ids'].shape
        inputs.append(batch)

    results = []
    for i in tqdm(range(eval_step), desc='Evaluating'):
        batch = inputs[i]
        batch['input_ids'] = batch['input_ids'].to(model.device)
        batch['attention_mask'] = batch['attention_mask'].to(model.device)
        bsz = batch.pop('bsz')
        input_len = batch.pop('input_len')

        with torch.no_grad():
            outputs = model.generate(**batch,
                                     max_new_tokens=512, pad_token_id=tokenizer.eos_token_id, stopping_criteria=stopping_criteria_list)
        output_text = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
        model_answer = [extract_predicted_answer(text.split("Q:")[0].split("</s>")[0].split("<|im_end|>")[0].strip()) for text in output_text]
        correct = [a == g for a, g in zip(model_answer, ground_truth_answer[i*batch_size:i*batch_size + bsz])]
        results += correct

    cnt = 0
    for result in results:
        if result:
            cnt += 1
    total = len(results)
    print(f"Accuracy: {cnt} / {total} = {cnt / total :.4f}")
    accuracy = cnt / total

    return accuracy

def eval_math(args, model, tokenizer):
    accuracy = evaluation(args, model, tokenizer)

    return accuracy


if __name__ == '__main__':
    evaluation()

