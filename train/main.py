from torch.utils.data import ConcatDataset, DataLoader
import torch.distributed as dist

import torch
import numpy as np
import json


from corpus import get_processor, RandomSampleCorpus
from latent_infer.misc import get_model_and_tokenizer, get_env_conf, get_torch_dtype, get_optimizer_and_lr_adjuster

import argparse, random, numpy, os
from itertools import chain
from functools import partial


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


def collate_fn(batch, train_rear_tokens=None):
    input_ids = [x.get('input_ids') for x in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.int64)

    labels = torch.zeros_like(input_ids)
    labels[..., :-1] = input_ids[..., 1:]
    labels[:, :-train_rear_tokens] = -100

    input_ids = input_ids[..., :-1]
    labels = labels[..., :-1]


    return dict(
        input_ids=input_ids.ravel().tolist(),
        labels=labels.ravel().tolist()
        )


def seed_everything(seed):
    torch.random.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)


def backend_setup():
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)


def backend_cleanup():
    dist.destroy_process_group()


def copy_gradients(params, grads):
    # copy gradients
    start, end = 0, 0
    for param in params:
        end = param.numel() + start
        param.grad.data.copy_(grads[start:end].reshape_as(param.grad.data))
        start = end


def compute_gradient(args, model, batch):
    local_rl_grads = []
    local_lm_grads = []
    local_rewards = []


    for _ in range(args.n_samples_per_gpu):
        print(f"{dist.get_rank}-ok")

        input_ids, labels = batch['input_ids'], batch['labels']

        rl_grads, lm_grads, rewards = model.sample(input_ids, labels)

        local_rl_grads.append(rl_grads)
        local_lm_grads.append(lm_grads)
        local_rewards.append(rewards)


    local_rl_grads = torch.stack(local_rl_grads, dim=0)
    local_lm_grads = torch.stack(local_lm_grads, dim=0)
    local_rewards = torch.cat(local_rewards)

    # local_rl_grads: (n_samples, n_elem)
    # local_lm_grads: (n_samples, n_elem)
    # local_rewards: (n_samples)

    assert local_rl_grads.ndim == 2 and local_lm_grads.ndim == 2 and local_rewards.ndim == 1
    assert local_rl_grads.shape[0] == local_lm_grads.shape[0] == local_rewards.shape[0]
        
    global_rl_grads = [torch.empty_like(local_rl_grads) for _ in range(dist.get_world_size())]
    global_lm_grads = [torch.empty_like(local_lm_grads) for _ in range(dist.get_world_size())]
    global_rewards = [torch.empty_like(local_rewards) for _ in range(dist.get_world_size())]

    dist.all_gather(global_rl_grads, local_rl_grads)
    dist.all_gather(global_lm_grads, local_lm_grads)
    dist.all_gather(global_rewards, local_rewards)

    global_rl_grads = torch.cat(global_rl_grads, dim=0)
    global_lm_grads = torch.cat(global_lm_grads, dim=0).mean()
    global_rewards = torch.cat(global_rewards).unsqueeze(-1)

    global_rewards = (global_rewards - global_rewards.mean()) / global_rewards.std()
    global_rl_grads = (global_rl_grads * global_rewards).mean(0)

    return torch.cat([global_rl_grads, global_lm_grads])



if __name__ == '__main__':


    backend_setup()


    parser = argparse.ArgumentParser()
    parser.add_argument("--env_conf", type=str, required=True)
    parser.add_argument("--n_samples_per_gpu", type=int, default=32)
    parser.add_argument("--train_rear_tokens", type=int, default=None)
    args = parser.parse_args()

    

    env_conf = get_env_conf(args.env_conf)
    env_conf['model']['device_map'] = {"": dist.get_rank()}
    dtype = get_torch_dtype(env_conf['model']['model_dtype'])


    # load model
    seed_everything(0)
    model, tokenizer = get_model_and_tokenizer(**env_conf['model'])
    seed_everything(dist.get_rank())


    params = model.ft_params()
    optimizer, lr_adjuster = get_optimizer_and_lr_adjuster(**env_conf['train'], params=params)


    model.enable_fsdp()


    # constraits
    assert args.n_samples_per_gpu % dist.get_world_size() == 0, f"argument `--n_samples` must be divisible by the number of GPUs"


    # build dataset
    if dist.get_rank() == 0:
        corpus = build_dataset(env_conf, tokenizer)
    dist.barrier()
    if dist.get_rank() != 0:
        corpus = build_dataset(env_conf, tokenizer)

    loader = DataLoader(
        corpus, 
        batch_size=1, 
        collate_fn=partial(collate_fn, train_rear_tokens=args.train_rear_tokens))


    for step, batch in enumerate(loader):
        lr_adjuster(step=step)
        optimizer.zero_grad()

        with model.no_sync():
            gradient = compute_gradient(args, model, batch)

        copy_gradients(params, gradient)
        optimizer.step()


        # if step % 100 == 0 and dist.get_rank() == 0:
        #     print(
        #         f"step-{step:<5d} | "
        #         f"loss_baseline: {loss_baseline.item():>.3f} | "
        #         f"loss: {np.mean(losses):>.3f} | "
        #         f"ratio: {np.mean(ratios):>.3f}", 
        #         flush=True)


    if dist.get_rank() == 0:
        model.save_checkpoint()


    backend_cleanup()
