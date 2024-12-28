from myabc.misc import get_model_and_tokenizer, get_env_conf
from corpus import get_processor, LazyRandomSampleCorpus
import argparse
from copy import deepcopy
import torch
import json


def build_dataset(env_conf, tokenizer):
    proc = get_processor(env_conf['eval']['config'], tokenizer)
    corp = LazyRandomSampleCorpus(env_conf['eval']['file_path'], proc, max_instance=env_conf['eval']['max_instance'])
    return corp


def colored_text(text, r, g, b):
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"


def gradient_color(string, x):
    if not (0 <= x <= 1):
        raise ValueError("Input must be between 0 and 1")
    if x <= 0.5:
        ratio = x / 0.5
        r = int(0 + (255 - 0) * ratio)
        g = 255
        b = 0
    else:
        ratio = (x - 0.5) / 0.5
        r = 255
        g = int(255 - (255 - 0) * ratio)
        b = 0
    return colored_text(string, r, g, b)


def filter_valid(x_list):
    return list(filter(lambda x: x is not None, x_list))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_conf", type=str)
    parser.add_argument("--trainable_layers", type=str, default="[0,1,2,3]")
    parser.add_argument("--trainable_tokens", type=int, default=16)
    parser.add_argument("--merge_method", type=str, default='avg')
    args = parser.parse_args()


    args.trainable_layers = json.loads(args.trainable_layers)
    env_conf = get_env_conf(args.env_conf)
    model, tokenizer = get_model_and_tokenizer(**env_conf['model'])
    
    corpus = build_dataset(env_conf, tokenizer)


    for data in corpus:
        input_ids = torch.tensor(data['input_ids'], dtype=torch.int64).unsqueeze(0)
        device = next(iter(model.parameters())).device
        input_ids = input_ids.to(device)

        labels = input_ids.clone()


        with torch.no_grad():
            loss, logits, masks = model(
                input_ids=input_ids, 
                labels=labels,
                return_logits=args.trainable_layers,
                deterministic_pruning=args.trainable_layers,
                trainable_tokens=args.trainable_tokens,
                merge_method=args.merge_method)


        with torch.no_grad():
            loss_baseline, _, _ = model(
                input_ids=input_ids, 
                labels=labels,
                trainable_tokens=args.trainable_tokens)


        print(f"loss-{loss.item()}, loss-baseline-{loss_baseline.item()}")
        print()


        input_ids = input_ids.ravel()
        cumulative_mask = torch.full_like(filter_valid(masks)[0], fill_value=False).ravel()

        num_pruned = 0
        for idx, (p, m) in enumerate(zip(logits, masks)):
            if p is not None:
                p = p.ravel().sigmoid()
                m = m.ravel()
                num_pruned += m.int().sum().item()

                cumulative_mask = torch.masked_scatter(cumulative_mask, cumulative_mask == False, m)
                
                print(f"layer-{idx}, num_pruned-{num_pruned}")
                print(tokenizer.decode(input_ids[:1]), end='')
                
                for x, i in zip(cumulative_mask, input_ids[1:]):
                    print(gradient_color(tokenizer.decode(i) + ' ' if i != 13 else '\\n', int(x.item())), end='')

                print()

        print('\n')
        print('=' * 40)
