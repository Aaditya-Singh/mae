python -m torch.distributed.launch --master_port=1324 main_finetune.py \
    --eval --batch_size 16 \
    --model deit_base \
    --resume pretrained/msn/vitb16_100ep_ft.pth \
    --data_path /srv/share4/datasets/ImageNet/imagenet/