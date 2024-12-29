CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --num_processes 2 \
    --main_process_port 29500 \
    train.py --config configs/train/train.yaml