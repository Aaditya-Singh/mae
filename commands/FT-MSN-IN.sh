Dataset=inet
SS=none
FT=full
Eval=IN1k
Model=msn_vitb16
Type=lpft

python submitit_finetune.py \
    --dist_eval --partition short --gpu-type a40 \
    --nodes 1 --ngpus 4 --timeout 2880 \
    --model deit_base --cls_token \
    --finetune pretrained/msn/msn_vitb16_inet_lineval_full.pth.tar \
    --epochs 20 --batch_size 32 --accum_iter 8 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --data_path /srv/share4/datasets/ImageNet/imagenet/ \
    --job_dir logs/submitit/${Model}/${Type}_${SS}/${FT}-${Eval}.log