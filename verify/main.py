import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

import argparse, random, numpy, json
from pygments.console import colorize

from corpus import get_processor, RandomSampleCorpus
from latent_infer.misc import (
    get_model_and_tokenizer,
    get_env_conf, 
    get_torch_dtype)


def build_dataset(env_conf, tokenizer):
    sum_partition = 0

    num_iters = env_conf['train']['train_iters']
    corpus = []
    for info in env_conf['train']['corpus']:
        sum_partition += info['partition']
        num_instance = int(info['partition'] * num_iters)

        proc = get_processor(info['conf'], tokenizer)
        corp = RandomSampleCorpus(info['data'], proc, max_instance=num_instance, use_cache=True)
        corpus.append(corp)

    assert sum_partition == 1
    return ConcatDataset(corpus)


def collate_fn(batch):
    assert len(batch) == 1

    input_ids = batch[0]['input_ids']
    input_ids = torch.tensor(input_ids, dtype=torch.int64, device='cuda').unsqueeze(0)

    labels = batch[0]['labels']
    labels = torch.tensor(labels, dtype=torch.int64, device='cuda').unsqueeze(0)

    return dict(
        input_ids=input_ids,
        labels=labels)


def seed_everything(seed):
    torch.random.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)


def copy_kv_cache(kv_cache):
    kv_cache_copy = []
    for i in range(len(kv_cache)):
        k = kv_cache[i][0].detach()
        v = kv_cache[i][1].detach()
        kv_cache_copy.append([k,v])
    return kv_cache_copy
    

class MulData(Dataset):
    def __init__(self, path, num_data):
        
        self.data = []
        with open(path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
                if len(self.data) >= num_data:
                    break

    def __len__(self):
        return self.data.__len__()
    

    def __getitem__(self, i):
        return self.data[i]
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_conf", type=str, required=True)
    parser.add_argument("--num_cot_tokens", type=int, default=3)
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()


    env_conf = get_env_conf(args.env_conf)
    env_conf['model']['device_map'] = {"": 0}
    dtype = get_torch_dtype(env_conf['model']['model_dtype'])


    # load model
    seed_everything(0)
    model, tokenizer = get_model_and_tokenizer(**env_conf['model'])


    # build dataset
    corpus = MulData(args.data_path, 8)
    loader = DataLoader(
        corpus, 
        batch_size=1, 
        collate_fn=collate_fn)


    for batch in loader:
        input_ids = batch['input_ids']
        labels = batch['labels']
        skip = (labels == -100).count_nonzero()

        # pre-fill first token
        with torch.no_grad():
            inputs = dict(
                input_ids=input_ids[:,:skip],
                input_embeds=None,
                mix_states=None,
                kv_cache=None)
            outputs = model(**inputs)

        print(colorize("yellow", '\n' + tokenizer.decode(input_ids.ravel().tolist() + labels.ravel().tolist()[-1:]).replace('$\times$', ' * ').replace('=', ' = ')))

        for i in range(skip, input_ids.shape[-1]):

            # ordinal infer
            with torch.no_grad():
                inputs = dict(
                    input_ids=input_ids[:,i:i+1],
                    input_embeds=None, 
                    mix_states=None,
                    kv_cache=outputs['kv_cache'])
                outputs = model(**inputs)

            mix_states = outputs['hidden_states']
            kv_cache_bkp = copy_kv_cache(outputs['kv_cache'])

            # accumulate loss
            logits = outputs['logits'].flatten(0,1)
            label = labels[:, i:i+1].ravel()
            loss_value = torch.nn.functional.cross_entropy(logits, label)
            
            print(f"baseline: {loss_value}")

            # latent infer
            for j in range(args.num_cot_tokens):

                with torch.no_grad():
                    inputs = dict(
                        input_ids=None,
                        input_embeds=outputs['latent_states'],
                        mix_states=mix_states,
                        kv_cache=outputs['kv_cache'])
                    outputs = model(**inputs)

                    # accumulate loss
                    logits = outputs['logits'].flatten(0,1)
                    label = labels[:, i:i+1].ravel()
                    loss_value = torch.nn.functional.cross_entropy(logits, label)
                    print(f"{j}: {loss_value}")

            outputs['kv_cache'] = kv_cache_bkp

            with torch.no_grad():
                logits = outputs['logits'].flatten(0,1)
                label = labels[:, i:i+1].ravel()
                loss_value = torch.nn.functional.cross_entropy(logits, label)


        # compute loss baseline
        with torch.no_grad():
            inputs = dict(
                input_ids=input_ids,
                input_embeds=None,
                mix_states=None,
                kv_cache=None)
            outputs = model(**inputs)

            labels[:, :skip] = -100
            baseline = torch.nn.functional.cross_entropy(outputs['logits'].flatten(0,1), labels.ravel())
