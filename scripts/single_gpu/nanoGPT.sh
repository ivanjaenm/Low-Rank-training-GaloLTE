# nanoGPT, 
# replace two lines below:
#8-bit GaLore-Adam, single GPU
# 22.72G, 0.37s/it
torchrun --nproc_per_node 1 torchrun_main.py \
    --model_config configs/llama_1b.json \
    --lr 0.005 \
    --galore_scale 0.25 \
    --rank 1024 \
    --update_proj_gap 500 \
    --batch_size 1 \
    --total_batch_size 512 \
    --num_training_steps 5 \
    --warmup_steps 2 \
    --weight_decay 0 \
    --grad_clipping 1.0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --single_gpu \
    --optimizer galore_adafactor \
    --save_dir /dev/shm/checkpoints/llama_1b