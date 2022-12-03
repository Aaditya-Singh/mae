# TODO: save logs in a file
SS=none
FT=full
Eval=IN1k
Model=msn_vitb16_ft
Type=ft

python -m torch.distributed.launch --master_port=1234 main_finetune.py \
    --eval --batch_size 16 \
    --model vit_base_patch16 \
    --resume pretrained/msn/vitb16_100ep_ft.pth \
    --data_path /srv/share4/datasets/ImageNet/imagenet/ \
    # --log_dir logs/${Model}/${Type}_${SS}/${FT}-${Eval}.log