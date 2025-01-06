import os
import argparse
import json
import random
import transformers



def generate_n_digit_number(n):
    lower = 10**(n-1) if n > 1 else 0
    upper = 10**n - 1
    return random.randint(lower, upper)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_data", type=int, default=65536)
    parser.add_argument("--num_digits", type=int, default=2)
    parser.add_argument("--tokenizer", type=str, default='unsloth/Qwen2.5-0.5B')
    args = parser.parse_args()

    
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
    path = f"data/mul/mul{args.num_digits}.jsonl"

    if os.path.exists(path):
        os.remove(path)


    for _ in range(args.num_data):

        a = generate_n_digit_number(args.num_digits)
        b = generate_n_digit_number(args.num_digits)
        c = a * b

        question = f"{a}$\times${b}="
        answer = f"{c}"

        question_ids = tokenizer(question, add_special_tokens=False).input_ids
        answer_ids = tokenizer(answer, add_special_tokens=False).input_ids


        input_ids = question_ids + answer_ids
        labels = [-100] * (len(question_ids) - 1) + answer_ids + [-100]

        input_ids = input_ids[:-1]
        labels = labels[:-1]

  
        data = dict(
            input_ids=input_ids,
            labels=labels)

        with open(path, 'a+') as f:
            f.write(json.dumps(data) + '\n')
