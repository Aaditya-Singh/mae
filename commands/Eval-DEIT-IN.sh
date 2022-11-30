# TODO: save logs in a file
# SS=subsets1
FT=IN1k
Eval=IN_S
Model=DEIT_ViTB_16
Type=Frozen

python -m torch.distributed.launch --master_port=1234 main_finetune.py \
    --eval --batch_size 16 \
    --model deit_base \
    --resume pretrained/deit/deit_base_patch16.pth \
    --data_path /srv/share4/datasets/ImageNet/ \
    # --log_dir logs/${Model}/${Type}_${SS}/${FT}-${Eval}.log \
  