export CUDA_VISIBLE_DEVICES=0

python visualize/visualize.py \
    --env_conf visualize/llama2-7b.json \
    --trainable_layers "[0,1,2]" \
    --trainable_tokens 16