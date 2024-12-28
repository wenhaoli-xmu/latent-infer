torchrun \
    --nnodes 1 \
    --nproc_per_node 4 \
    train/main.py \
    --env_conf train/qwen2.5-0.5b.json \
    --n_samples 64 \
    --train_rear_tokens 64