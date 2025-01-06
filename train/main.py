import torch
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler, Dataset
import torch.distributed as dist

import argparse, random, numpy, os, json

from corpus import get_processor, RandomSampleCorpus
from latent_infer.misc import (
    get_model_and_tokenizer,
    get_env_conf, 
    get_torch_dtype, 
    get_optimizer_and_lr_adjuster,
    History,
    GradientAccumulator)


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


# def collate_fn(batch):
#     input_ids = batch[0]['input_ids']
#     input_ids = torch.tensor(input_ids, dtype=torch.int64, device='cuda').unsqueeze(0)

#     labels = torch.zeros_like(input_ids)
#     labels[..., :-1] = input_ids[..., 1:]

#     input_ids = input_ids[..., :-1]
#     labels = labels[..., :-1]

#     return dict(
#         input_ids=input_ids,
#         labels=labels)


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


def backend_setup():
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)


def backend_cleanup():
    dist.destroy_process_group()


def copy_kv_cache(kv_cache):
    kv_cache_copy = []
    for i in range(len(kv_cache)):
        k = kv_cache[i][0].detach()
        v = kv_cache[i][1].detach()
        kv_cache_copy.append([k,v])
    return kv_cache_copy


class Loss:
    def __init__(self):
        self.loss_for_backward = 0
        self.loss_for_comparison = 0
        self.num_loss_for_bwd = 0
        self.num_loss_for_cmp = 0

    def update_inside(self, loss):
        self.loss_for_backward += loss
        self.num_loss_for_bwd += 1

    def update_outside(self, loss):
        self.loss_for_comparison += loss
        self.num_loss_for_cmp += 1

    def backward(self):
        if self.num_loss_for_bwd == 0:
            return
        (self.loss_for_backward / self.num_loss_for_bwd).backward()

    def item(self):
        return self.loss_for_comparison.item() / self.num_loss_for_cmp
    

class MulData(Dataset):
    def __init__(self, path):
        
        self.data = []
        with open(path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return self.data.__len__()
    

    def __getitem__(self, i):
        return self.data[i]




if __name__ == '__main__':


    backend_setup()


    parser = argparse.ArgumentParser()
    parser.add_argument("--env_conf", type=str, required=True)
    # parser.add_argument("--prob", type=float, default=0.5)
    # parser.add_argument("--last_n", type=int, default=16)
    parser.add_argument("--num_cot_tokens", type=int, default=3)
    parser.add_argument("--num_accum_steps", type=int, default=1)
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()


    env_conf = get_env_conf(args.env_conf)
    env_conf['model']['device_map'] = {"": dist.get_rank()}
    dtype = get_torch_dtype(env_conf['model']['model_dtype'])


    # load model
    seed_everything(0)
    model, tokenizer = get_model_and_tokenizer(**env_conf['model'])
    seed_everything(dist.get_rank())


    params = model.ft_params()
    optimizer, lr_adjuster = get_optimizer_and_lr_adjuster(
        **env_conf['train'], 
        params=params)
    optimizer = GradientAccumulator(optimizer, params, accum_steps=args.num_accum_steps)


    # build dataset
    corpus = MulData(args.data_path)

    sampler = DistributedSampler(
        corpus, 
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=False)

    loader = DataLoader(
        corpus, 
        batch_size=1, 
        collate_fn=collate_fn,
        sampler=sampler)
    
    sampler.set_epoch(0)


    history = History()


    for step, batch in enumerate(loader):

        lr_adjuster(step=step)
        optimizer.zero_grad()

        input_ids = batch['input_ids']
        labels = batch['labels']

        skip = (labels == -100).count_nonzero()
        # skip = input_ids.shape[-1] - args.last_n

        # pre-fill first token
        with torch.no_grad():
            inputs = dict(
                input_ids=input_ids[:,:skip],
                input_embeds=None,
                mix_states=None,
                kv_cache=None)
            outputs = model(**inputs)


        loss = Loss()

        for i in range(skip, input_ids.shape[-1]):

            # ordinal infer
            inputs = dict(
                input_ids=input_ids[:,i:i+1],
                input_embeds=None, 
                mix_states=None,
                kv_cache=outputs['kv_cache'])
            outputs = model(**inputs)

            mix_states = outputs['hidden_states']
            kv_cache_bkp = copy_kv_cache(outputs['kv_cache'])

            # latent infer
            for _ in range(args.num_cot_tokens):
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
                loss.update_inside(loss_value)

            outputs['kv_cache'] = kv_cache_bkp

            with torch.no_grad():
                logits = outputs['logits'].flatten(0,1)
                label = labels[:, i:i+1].ravel()
                loss_value = torch.nn.functional.cross_entropy(logits, label)

            loss.update_outside(loss_value)
            

        # backward propagation
        loss.backward()

        for param in params:
            dist.all_reduce(param.grad.data)
            param.grad.data /= dist.get_world_size()

        optimizer.step()


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

        history.update(loss.item(), baseline.item())


    if dist.get_rank() == 0:
        history.summary()
        model.save_checkpoint()


    backend_cleanup()
