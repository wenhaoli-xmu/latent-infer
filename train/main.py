import torch
from torch.utils.data import ConcatDataset, DataLoader
import torch.distributed as dist

import argparse, random, numpy, os
from functools import partial

from corpus import get_processor, RandomSampleCorpus
from latent_infer.misc import (
    get_model_and_tokenizer,
    get_env_conf, 
    get_torch_dtype, 
    get_optimizer_and_lr_adjuster)


def filter_valid(x_list):
    return list(filter(lambda x: x is not None, x_list))


def zero_grad(rl_params, lm_params):
    for param in rl_params + lm_params:
        if param.grad is not None:
            param.grad.data.zero_()


def collect_grads(rl_params, lm_params):

    rl_grads = []
    lm_grads = []

    for param in rl_params:
        if param.grad is not None:
            rl_grads.append(param.grad.data.ravel())
        else:
            rl_grads.append(torch.zeros_like(param.data).ravel())

    
    for param in lm_params:
        if param.grad is not None:
            lm_grads.append(param.grad.data.ravel())
        else:
            lm_grads.append(torch.zeros_like(param.data).ravel())

    return rl_grads, lm_grads



def enable_fsdp(model):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    class_type = type(model._get_layers()[0])

    my_auto_wrap_policy = partial(
        transformer_auto_wrap_policy, 
        transformer_layer_cls=set([class_type]))

    return FSDP(
        module=model, 
        auto_wrap_policy=my_auto_wrap_policy)


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
    torch.cuda.set_device(local_rank)



def backend_cleanup():
    dist.destroy_process_group()


def copy_gradients(params, grads):
    # copy gradients
    start, end = 0, 0
    for param in params:
        end = param.numel() + start
        param.grad = grads[start:end].reshape_as(param.data)
        start = end


def my_slice(x, start, end):
    return torch.tensor(x[start:end], dtype=torch.int64).unsqueeze(0).cuda()


def compute_reward(lm_loss, ratio, alpha):
    reward = -lm_loss - alpha * ratio
    reward = torch.tensor(reward, dtype=torch.bfloat16, device='cuda')
    return reward


def sample(args, model, input_ids: list, labels: list, rl_params, lm_params):

    """
    假设input-ids和labels是已经错开的了
    """

    assert isinstance(input_ids, list)
    assert isinstance(labels, list)
    assert labels[0] == -100


    zero_grad(rl_params, lm_params)


    rl_losses, lm_losses, flags = [], [], []


    # NOTE: pre-filling phase
    pos = 0
    while labels[pos] == -100:
        pos += 1

    with torch.no_grad():
        outputs = model(
            input_ids=my_slice(input_ids, 0, pos),
            label=None,
            kv_cache=None)
        outputs['flag'] = None


    # NOTE: decoding phase
    while pos < len(labels):
        if outputs['flag'] in (None, False):
            outputs = model(
                input_ids=my_slice(input_ids, pos, pos + 1),
                label=labels[pos],
                kv_cache=outputs['kv_cache'])
        
        elif outputs['flag'] is True:
            outputs = model(
                input_ids=None,
                label=None,
                kv_cache=outputs['kv_cache'])

        lm_losses.append(outputs['loss'])
        rl_losses.append(outputs['nll'])
        flags.append(outputs['flag'])
        pos += 1


    # backward
    lm_losses = filter_valid(lm_losses)
    lm_loss = torch.stack(lm_losses).mean()
    rl_loss = torch.stack(rl_losses).mean()
    (lm_loss + rl_loss).backward()


    # ratio & reward
    ratio = sum(flags) / len(flags)
    reward = compute_reward(lm_loss.item(), ratio, args.alpha)


    # collect gradients
    rl_grads, lm_grads = collect_grads(rl_params, lm_params)


    return dict(
        rl_grads=torch.cat(rl_grads, dim=0),
        lm_grads=torch.cat(lm_grads),
        reward=reward,
        ratio=ratio,
        lm_loss=lm_loss.item())



def compute_baseline(args, model, batch):
    input_ids, labels = batch['input_ids'], batch['labels']

    with torch.no_grad():
        _, logits, _ = model.model(
            input_ids=my_slice(input_ids, 0, len(input_ids)),
            kv_cache=None,
            reduce_logits=False)

        labels = torch.tensor(labels, dtype=torch.int64, device='cuda')
        logits = logits.squeeze(0)
        loss = torch.nn.functional.cross_entropy(logits, labels)

    return loss.item()



def compute_gradient(args, model, batch, rl_params, lm_params):
    local_rl_grads = []
    local_lm_grads = []
    local_rewards = []
    local_ratios = []
    local_lm_losses = []


    for _ in range(args.n_samples // dist.get_world_size()):

        input_ids, labels = batch['input_ids'], batch['labels']

        outputs = sample(
            args=args, 
            model=model,
            input_ids=input_ids, 
            labels=labels, 
            rl_params=rl_params, 
            lm_params=lm_params)

        local_rl_grads.append(outputs['rl_grads'])
        local_lm_grads.append(outputs['lm_grads'])
        local_rewards.append(outputs['reward'])
        local_ratios.append(outputs['ratio'])
        local_lm_losses.append(outputs['lm_loss'])


    local_rl_grads = torch.stack(local_rl_grads, dim=0)
    local_lm_grads = torch.stack(local_lm_grads, dim=0)
    local_rewards = torch.stack(local_rewards)
    local_ratio_avg = torch.tensor(sum(local_ratios) / len(local_ratios), device='cuda')
    local_lm_loss_avg = torch.tensor(sum(local_lm_losses) / len(local_lm_losses), device='cuda')

    assert local_rl_grads.ndim == 2 and local_lm_grads.ndim == 2 and local_rewards.ndim == 1
    assert local_rl_grads.shape[0] == local_lm_grads.shape[0] == local_rewards.shape[0]
        
    global_rl_grads = [torch.empty_like(local_rl_grads) for _ in range(dist.get_world_size())]
    global_lm_grads = [torch.empty_like(local_lm_grads) for _ in range(dist.get_world_size())]
    global_rewards = [torch.empty_like(local_rewards) for _ in range(dist.get_world_size())]
    global_ratio_avg = [torch.empty_like(local_ratio_avg) for _ in range(dist.get_world_size())]
    global_lm_loss_avg = [torch.empty_like(local_lm_loss_avg) for _ in range(dist.get_world_size())]

    dist.all_gather(global_rl_grads, local_rl_grads)
    dist.all_gather(global_lm_grads, local_lm_grads)
    dist.all_gather(global_rewards, local_rewards)
    dist.all_gather(global_ratio_avg, local_ratio_avg)
    dist.all_gather(global_lm_loss_avg, local_lm_loss_avg)

    global_rl_grads = torch.cat(global_rl_grads, dim=0)
    global_lm_grads = torch.cat(global_lm_grads, dim=0).mean(0)
    global_rewards = torch.cat(global_rewards).unsqueeze(-1)
    global_ratio_avg = (sum(global_ratio_avg) / len(global_ratio_avg)).item()
    global_lm_loss_avg = (sum(global_lm_loss_avg) / len(global_lm_loss_avg)).item()

    global_rewards = (global_rewards - global_rewards.mean()) / global_rewards.std()
    global_rl_grads = (global_rl_grads * global_rewards).mean(0)

    return dict(
        rl_grads=global_rl_grads,
        lm_grads=global_lm_grads,
        ratio=global_ratio_avg,
        lm_loss=global_lm_loss_avg)



if __name__ == '__main__':


    backend_setup()


    parser = argparse.ArgumentParser()
    parser.add_argument("--env_conf", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=32)
    parser.add_argument("--train_rear_tokens", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()


    env_conf = get_env_conf(args.env_conf)
    env_conf['model']['device_map'] = {"": dist.get_rank()}
    dtype = get_torch_dtype(env_conf['model']['model_dtype'])


    # load model
    seed_everything(0)
    model, tokenizer = get_model_and_tokenizer(**env_conf['model'])
    seed_everything(dist.get_rank())


    rl_params, lm_params = model.rl_params(), model.lm_params()
    optimizer, lr_adjuster = get_optimizer_and_lr_adjuster(
        **env_conf['train'], 
        rl_params=rl_params,
        lm_params=lm_params)


    # model = enable_fsdp(model)
    # torch.cuda.empty_cache()


    # constraits
    assert args.n_samples % dist.get_world_size() == 0, f"argument `--n_samples` must be divisible by the number of GPUs"


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

        baseline = compute_baseline(args, model, batch)

        outputs = compute_gradient(args, model, batch, rl_params, lm_params)

        copy_gradients(rl_params, outputs['rl_grads'])
        copy_gradients(lm_params, outputs['lm_grads'])
        optimizer.step()


        if dist.get_rank() == 0:
            print(
                f"step-{step:<5d} | "
                f"baseline-{baseline:>.3f} | "
                f"loss: {outputs['lm_loss']:>.3f} | "
                f"ratio: {outputs['ratio']:>.3f}", 
                flush=True)


    if dist.get_rank() == 0:
        model.save_checkpoint()


    backend_cleanup()
