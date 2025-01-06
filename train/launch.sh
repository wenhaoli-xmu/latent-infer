torchrun \
    --nnodes 1 \
    --nproc_per_node 8 \
    train/main.py \
    --env_conf train/qwen2.5-0.5b.json \
    --data_path data/mul/mul2.jsonl \
    --num_accum_steps 8 \
    --num_cot_tokens 4