torchrun \
    --nnodes 1 \
    --nproc_per_node 4 \
    --master_port 11001 \
    train/main.py \
    --env_conf train/qwen2.5-3b.json \
    --data_path data/mul/mul2.jsonl \
    --num_accum_steps 16 \
    --num_cot_tokens 2 \
    --id level2_latent2