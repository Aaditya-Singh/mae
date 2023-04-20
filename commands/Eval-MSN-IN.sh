python -m torch.distributed.launch --master_port=1234 main_finetune.py \
    --eval --batch_size 16 \
    --model deit_base --cls_token \
    --resume pretrained/msn/vitb16_100ep_ft.pth \
    --data_path ../datasets/ImageNet/imagenet/
