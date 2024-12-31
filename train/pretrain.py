import torch
import torch.utils
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import argparse, random, numpy, os
from functools import partial

from corpus import get_processor, RandomSampleCorpus
from latent_infer.misc import (
    get_model_and_tokenizer,
    get_env_conf, 
    get_torch_dtype, 
    get_optimizer_and_lr_adjuster)


from pygments.console import colorize
from copy import deepcopy
import math


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


def enable_fsdp(model):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    class_type = type(model.get_model().model.layers[0])

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


def make_tensor(batch):
    length = [len(data['input_ids']) for data in batch]
    max_length = max(length)

    for i in range(len(batch)):
        batch[i]['input_ids'] = batch[i]['input_ids'] + [-100] * (max_length - length[i])
        batch[i]['labels'] = batch[i]['labels'] + [-100] * (max_length - length[i])
        batch[i]['position_ids'] = batch[i]['position_ids'] + [-100] * (max_length - length[i])


    input_ids = torch.tensor([data['input_ids'] for data in batch])
    labels = torch.tensor([data['labels'] for data in batch])
    position_ids = torch.tensor([data['position_ids'] for data in batch])

    return dict(
        input_ids=input_ids,
        labels=labels,
        position_ids=position_ids)


def collate_fn(batch):
    original_batch = deepcopy(batch)
    for i, data in enumerate(original_batch):
        num_tokens = len(data['input_ids'])
        original_batch[i]['labels'] = deepcopy(data['input_ids'])
        original_batch[i]['labels'] = data['labels'][1:] + [-100]
        original_batch[i]['position_ids'] = list(range(num_tokens))

    for i, data in enumerate(batch):
        data['labels'] = deepcopy(data['input_ids'])
        data['labels'] = data['labels'][1:] + [-100]

        new_input_ids = []
        new_labels = []
        position_ids = []

        for j in range(len(data['input_ids'])):
            new_input_ids.append(data['input_ids'][j])
            new_labels.append(data['labels'][j])
            position_ids.append(j)

            rand_gaussian = abs(torch.randn(1).item())
            num_latent = int(math.ceil(rand_gaussian))

            new_input_ids += [-100] * num_latent
            new_labels += [-100] * num_latent
            position_ids += [-100] * num_latent

        batch[i]['input_ids'] = new_input_ids
        batch[i]['labels'] = new_labels
        batch[i]['position_ids'] = position_ids

    return dict(
        baseline=make_tensor(original_batch),
        latent_infer=make_tensor(batch))


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


def print_info(step, outputs_baseline, outputs, params):
    magnitude = 0
    magnitude_grad = 0
    for param in params:
        magnitude += param.data.abs().mean().item()
        magnitude_grad += param.grad.data.abs().mean().item()


    if dist.get_rank() == 0:
        if abs(outputs_baseline['loss'] - outputs['loss']) < 0.04:
            color = "yellow"
        elif outputs_baseline['loss'] > outputs['loss']:
            color = "green"
        else:
            color = "red"

        loss_info = "loss: " + colorize(color, f"{outputs['loss']:>.3f}")
        print(
            f"step-{step:<5d} | "
            f"baseline: {outputs_baseline['loss']:>.3f} | "
            f"{loss_info} | "
            f"magnitude(param): {magnitude:.5f} | "
            f"magnitude(grad): {magnitude_grad:.5f}",
            flush=True)
        

def visualize(batch, outputs_baseline, outputs):
    input_ids = batch['baseline']['input_ids'].ravel().tolist()
    input_ids = list(filter(lambda x: x != -100, input_ids))[:-1]
    input_ids = [tokenizer.decode(token) for token in input_ids]

    latent_loss = outputs['detailed_loss'].ravel().tolist()
    latent_loss = list(filter(lambda x: x != 0, latent_loss))
    base_loss = outputs_baseline['detailed_loss'].ravel().tolist()[:-1]

    res_loss = [x - y for x, y in zip(latent_loss, base_loss)]
    
    def _f(x):
        if abs(x) < 0.1:
            return 0.5
        elif x > 0:
            return 1
        elif x < 0:
            return 0
        
    res_loss_filtered = list(map(_f, res_loss))

    for x, y in zip(input_ids, res_loss_filtered):
        x = gradient_color(x, y)
        print(x.replace('\n', '\\n'), end='')


    print(end='\n\n')
    print(f"=" * 80)

    for x, y, z in zip(input_ids, res_loss_filtered, res_loss):
        x = gradient_color(x, y)
        print(x.replace('\n', '\\n') + f'({z:.1f})', end='')


def build_mask(batch):

    attention_mask = []

    for input_ids in batch['latent_infer']['input_ids']:
        input_ids = input_ids.ravel().tolist()

        length = len(input_ids)
        infinity = torch.finfo(torch.bfloat16).min
        x = torch.zeros((length, length), dtype=torch.bfloat16, device='cuda')
        x.fill_(infinity)
        x.triu_(1)

        for i in range(len(input_ids) - 1):
            if input_ids[i] != -100 and input_ids[i+1] == -100:
                for j in range(i):
                    if input_ids[j] == -100:
                        x[i+1:,j].fill_(infinity)

        attention_mask.append(x.reshape(1,1,length,length))

    return torch.cat(attention_mask, dim=0)


if __name__ == '__main__':


    backend_setup()


    parser = argparse.ArgumentParser()
    parser.add_argument("--env_conf", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--visualize", type=int, default=100)
    args = parser.parse_args()


    assert args.batch_size % dist.get_world_size() == 0


    env_conf = get_env_conf(args.env_conf)
    env_conf['model']['device_map'] = {"": dist.get_rank()}
    dtype = get_torch_dtype(env_conf['model']['model_dtype'])


    # load model
    seed_everything(0)
    model, tokenizer = get_model_and_tokenizer(**env_conf['model'])
    model.init_latent_embed(tokenizer.eos_token_id)
    seed_everything(dist.get_rank())

    params = model.ft_params()
    optimizer, lr_adjuster = get_optimizer_and_lr_adjuster(
        **env_conf['train'], 
        params=params)

    model = DDP(model)
    

    # dataset stuff
    if dist.get_rank() == 0:
        corpus = build_dataset(env_conf, tokenizer)
    dist.barrier()
    corpus = build_dataset(env_conf, tokenizer)
    sampler = DistributedSampler(corpus, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
    loader = DataLoader(
        corpus, 
        batch_size=args.batch_size // dist.get_world_size(), 
        collate_fn=collate_fn,
        sampler=sampler)
    sampler.set_epoch(0)


    for step, batch in enumerate(loader):
        lr_adjuster(step=step)
        optimizer.zero_grad()

        with torch.no_grad():
            outputs_baseline = model(**batch['baseline'])

        outputs = model(**batch['latent_infer'], attention_mask=build_mask(batch))
        outputs['loss'].backward()
        optimizer.step()

        print_info(step, outputs_baseline, outputs, params)
        
        # if (step + 1) % args.visualize == 0 and dist.get_rank() == 0:
        #     visualize(batch, outputs_baseline, outputs)
        #     import IPython
        #     IPython.embed()
        # dist.barrier()

    if dist.get_rank() == 0:
        model.save_checkpoint()


    backend_cleanup()
