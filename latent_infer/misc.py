import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .modifiers import get_modifier
from functools import partial
import os, math


from torch import distributed as dist





def get_torch_dtype(dtype: str):
    if dtype == 'fp16':
        return torch.float16
    elif dtype == 'fp32':
        return torch.float32
    elif dtype == 'bf16':
        return torch.bfloat16
    else:
        raise RuntimeError(f"Unknown dtype '{dtype}'")


def get_env_conf(env_conf: str):
    import json
    with open(env_conf, 'r') as f:
        env_conf = json.load(f)
    return env_conf


def get_model_and_tokenizer(
        model_name, 
        model_dtype, 
        model_method, 
        model_structure, 
        save_ckp, 
        load_ckp, 
        config, 
        device_map, 
        **kwargs
    ):

    from accelerate import dispatch_model
    token = os.environ['HF_ACCESS_TOKEN']

    if "tokenizer_name" in kwargs:
        tokenizer = AutoTokenizer.from_pretrained(
            kwargs.get('tokenizer_name'), 
            use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            use_fast=True)

    student_dtype = get_torch_dtype(model_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=student_dtype, 
        token=token, 
        device_map="auto" if device_map is None else None,
        trust_remote_code=True)
    modifier = get_modifier(model_method, model_structure)

    if modifier is not None:
        model = modifier(
            model,
            save_ckp=save_ckp,
            load_ckp=load_ckp,
            config=config)

    if device_map is not None:
        model.model = dispatch_model(model.model, device_map=device_map)

    return model, tokenizer


def lr_scheduler(epoch, total_epochs, warmup, plateau, max_lr, min_lr, restart=20):
    total_epochs /= restart
    epoch = epoch % total_epochs

    if epoch / total_epochs < warmup:
        partial = epoch / int(total_epochs * warmup)
        return partial * (max_lr - min_lr) + min_lr
    
    elif epoch / total_epochs < warmup + plateau:
        return max_lr
    
    else:
        epoch -= int(total_epochs * (warmup + plateau))
        total_epochs -= int(total_epochs * (warmup + plateau))
        cos_decay = 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))
        lr = (max_lr - min_lr) * cos_decay + min_lr
    
    return lr


def adjust_lr(optim, step, total, max_lr, min_lr, restart, warmup, plateau):
    for param_group in optim.param_groups:
        param_group['lr'] = lr_scheduler(
            step,
            total,
            warmup=warmup,
            plateau=plateau,
            max_lr=max_lr,
            min_lr=min_lr,
            restart=restart
        )

def get_optimizer_and_lr_adjuster(max_lr, train_iters, warmup, weight_decay, beta1, beta2, params, **kwargs):
    optim = torch.optim.AdamW(
        params=params,
        betas=[beta1, beta2],
        weight_decay=weight_decay
    )

    lr_adjuster = partial(adjust_lr, optim=optim, total=train_iters, max_lr=max_lr, min_lr=0, restart=1, warmup=warmup, plateau=0)

    return optim, lr_adjuster



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


class PolicyGradient:
    def __init__(self, params):
        self.params = params
        self._grad = None
        self._reset()


    def _normalize(self, rewards):
        # calculate reward
        _mean = rewards.mean()
        _std = rewards.std()
        return (rewards - _mean) / (1e-8 + _std)

    def _copy_grad(self):
        # copy gradients
        start, end = 0, 0
        for param in self.params:
            end = param.numel() + start
            param.grad.data.copy_(self._grad[start:end].reshape_as(param))
            start = end

    def _reset(self):
        self.local_rewards = []
        self.local_grads = []

    def update(self, sample_outputs, grad):
        self.local_rewards.append(sample_outputs['reward'])
        self.local_grads.append(grad)

    def step(self):
        rewards = torch.tensor(self.local_rewards, dtype=torch.bfloat16, device='cuda')
        grads = torch.stack(self.local_grads, dim=0)

        rewards_list = [torch.empty_like(rewards) for _ in range(dist.get_world_size())]
        grads_list = [torch.empty_like(grads) for _ in range(dist.get_world_size())]

        dist.all_gather(rewards_list, rewards)
        dist.all_gather(grads_list, grads)

        rewards = torch.cat(rewards_list).unsqueeze(-1)
        grads = torch.cat(grads_list, dim=0) * self._normalize(rewards)

        grad = grads.mean(0)
        self._grad = (self._grad + grad) if self._grad is not None else grad

        self._reset()

    def prepare(self):
        self._copy_grad()
        self._grad = None
