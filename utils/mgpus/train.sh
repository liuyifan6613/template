CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --main_process_port=8888 main.py \
      --with_tracking \
      --train_img '/home/yfliu/Dataset/set14/' \
      --train_gt '/home/yfliu/Dataset/set14/' \
      --test_img '/home/yfliu/Dataset/set14/' \
      --test_gt '/home/yfliu/Dataset/set14/' \
      --eval_every 1 \
      --output_dir './checkpoints' \
#      --resume_from_checkpoint \
      --lr 1e-2 \
      --num_epochs 1000 \
      --batch_size 32 \
      --image_size 256 \
      --test False