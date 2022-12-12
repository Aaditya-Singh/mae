# TODO: save logs in a file
SS=none
FT=full
Eval=ObjectNet
Model=msn_vitb16_ft
Type=ft

python -m torch.distributed.launch --master_port=1234 main_finetune.py \
    --eval --batch_size 16 \
    --model deit_base \
    --resume pretrained/msn/vitb16_100ep_ft.pth \
    --data_path /srv/share4/datasets/ImageNet/objectnet-1.0/images/ \
    # --log_dir logs/${Model}/${Type}_${SS}/${FT}-${Eval}.log