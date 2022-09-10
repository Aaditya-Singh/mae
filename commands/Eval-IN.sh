# TODO: save logs in a file
# SS=subsets1
FT=IN1k
Eval=IN_S
Model=MAE_FT_ViTB_16
Type=Frozen

python -m torch.distributed.launch --master_port=1234 main_finetune.py --eval \
    --resume pretrained/mae_finetuned_vit_base.pth --model vit_base_patch16 \
    --batch_size 16 --data_path /srv/datasets/ImageNet/imagenet-s/ \
    # --log_dir logs/${Model}/${Type}_${SS}/${FT}-${Eval}.log \
  