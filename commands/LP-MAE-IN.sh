Dataset=inet
SS=subsets1
FT=1imgs_class
Eval=IN1k
Model=mae_vitb16
Type=lp

python -m torch.distributed.launch --master_port=1234 --nproc_per_node=4 main_linprobe.py \
    --dist_eval --model vit_base_patch16 --cls_token \
    --finetune pretrained/mae/mae_pretrain_vit_base.pth \
    --epochs 90 --batch_size 128 --accum_iter 32 \
    --blr 0.1 --weight_decay 0.0 \
    --data_path /srv/share4/datasets/ImageNet/imagenet/ \
    --subset_file imagenet_${SS}/${FT}.txt \
    --log_dir logs/${Model}/${Type}_${SS}/${FT}-${Eval}.log    