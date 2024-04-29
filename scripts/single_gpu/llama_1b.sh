# LLaMA-7B, 8-bit GaLore-Adam, single GPU
# 22.72G, 0.37s/it
# relevant arguments:
# single GPU - or DDP
torchrun --nproc_per_node 1 torchrun_main.py \
    --model_config configs/llama_1b.json \
    --seed 0 \
    --single_gpu \
    --use_galore \
    --optimizer galore_adamw8bit \
    --optimizer_lr 0.005 \
    --galore_rank 1024 \
    --galore_update_proj_gap 500 \
    --galore_scale 0.25 \
    --galore_DDP \
    --batch_size 1 \
    --total_batch_size 512 \
    --num_training_steps 1 \
    --warmup_steps 2 \
    --weight_decay 0 \
    --grad_clipping 1.0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --eval_tokens 10000 \
    --save_dir /dev/shm/checkpoints/llama_1b
    #--use_LTE \
    #--LTE_rank 32 \
    #--LTE_alpha 4096 \
    #--LTE_num_heads 32 \
    