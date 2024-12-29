CUDA_VISIBLE_DEVICES=0 python -u inference.py \
--task sr \
--upscale 1 \
--ckpt /home/yfliu/Thesis_Work/deblur/deblur/DiffBasic2/weight/v1_general.pth \
--sd_ckpt /home/yfliu/Thesis_Work/deblur/deblur/DiffBasic2/weights/v2-1_512-ema-pruned.ckpt \
--cfg_scale 8 \
--noise_aug 0 \
--input ./img \
--output ./out 