python -m torch.distributed.launch --master_port=1432 main_finetune.py \
    --eval --batch_size 16 \
    --model vit_base_patch16 \
    --resume pretrained/mae/mae_finetuned_vit_base.pth \
    --data_path /srv/share4/datasets/ImageNet/imagenet-r/
  