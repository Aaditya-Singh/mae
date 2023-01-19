Dataset=inet
SS=none
FT=full
Eval=IN1k
Model=msn_vitb16
Type=lineval

python submitit_linprobe.py \
    --dist_eval --partition long --gpu-type a40 \
    --nodes 1 --ngpus 4 \
    --model deit_base --cls_token \
    --finetune pretrained/msn/vitb16_600ep.pth.tar \
    --epochs 90 --batch_size 128 --accum_iter 32 \
    --blr 0.1 --weight_decay 0.0 \
    --data_path /srv/share4/datasets/ImageNet/imagenet/ \
    --job_dir logs/submitit/${Model}/${Type}_${SS}/${FT}-${Eval}.log