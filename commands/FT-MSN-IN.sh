Dataset=inet
SS=none
FT=inet
Eval=IN1k
Model=msn_vitb16
Type=ft

python submitit_finetune.py \
    --dist_eval --partition long --gpu-type a40 \
    --nodes 1 --ngpus 4 --timeout 7200 \
    --model deit_base --cls_token \
    --finetune pretrained/msn/vitb16_600ep.pth.tar \
    --epochs 100 --batch_size 32 --accum_iter 8 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --data_path ../datasets/ImageNet/imagenet/ \
    --job_dir logs/submitit/${Model}/${Type}_${SS}/${FT}-${Eval}.log
