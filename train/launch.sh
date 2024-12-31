torchrun \
    --nnodes 1 \
    --nproc_per_node 8 \
    train/pretrain.py \
    --env_conf train/qwen2.5-0.5b.json \
    --batch_size 8