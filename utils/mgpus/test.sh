CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --main_process_port=8888 main.py \
      --test_img '/home/yfliu/Dataset/set14/' \
      --test_gt '/home/yfliu/Dataset/set14/' \
      --resume_from_checkpoint '/home/yfliu/InstaFlow/checkpoints/model_99' \
      --num_epochs 1000 \
      --batch_size 16 \
      --image_size 256 \
      --test True