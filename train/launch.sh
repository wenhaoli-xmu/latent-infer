torchrun \
    --nnodes 1 \
    --nproc_per_node 4 \
    train/main.py \
    --env_conf train/llama2-7b.json \
    --n_samples_per_gpu 8 \
    --train_rear_tokens 8