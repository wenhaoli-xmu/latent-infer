torchrun \
    --nnodes 1 \
    --nproc_per_node 4 \
    train/main.py \
    --env_conf train/qwen2.5-0.5b.json \
    --prob 0.5 \
    --last_n 16