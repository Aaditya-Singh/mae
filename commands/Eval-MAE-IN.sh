python -m torch.distributed.launch --master_port=1234 main_finetune.py \
    --eval --batch_size 16 \
    --model vit_base_patch16 \
    --resume pretrained/mae/mae_finetuned_vit_base.pth \
    --data_path ../datasets/ImageNet/imagenet/
