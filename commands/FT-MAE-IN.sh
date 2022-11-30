Dataset=inet
SS=subsets1
FT=1imgs_class
Eval=IN1k
Model=mae_vitb16
Type=ft

python -m torch.distributed.launch --master_port=1234 --nproc_per_node=4 main_finetune.py \
    --dist_eval --model vit_base_patch16 \
    --finetune pretrained/mae/mae_pretrain_vit_base.pth \
    --epochs 100 --batch_size 32 --accum_iter 8 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --data_path /srv/share4/datasets/ImageNet/imagenet/ \
    --subset_file imagenet_${SS}/${FT}.txt \
    --log_dir logs/${Model}/${Type}_${SS}/${FT}-${Eval}.log