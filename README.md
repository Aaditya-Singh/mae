## [Masked Autoencoders](https://arxiv.org/abs/2111.06377): An Unofficial PyTorch Implementation

<p align="center">
  <img src="https://user-images.githubusercontent.com/11435359/146857310-f258c86c-fde6-48e8-9cee-badd2b21bd2c.png" width="480">
</p>

This repo is a modification of the [official MAE repo](https://github.com/facebookresearch/mae). Please follow their [instructions](https://github.com/facebookresearch/mae#masked-autoencoders-a-pytorch-implementation) for installing any additional or conflicting requirements.

## Full fine-tuning on ImageNet

The bash command used for fine-tuning on full ImageNet can be found [here](https://github.com/Aaditya-Singh/mae/blob/main/commands/FT-MSN-IN.sh). We summarize some of the important flags for experimentation below:

* `finetune`: Specifies the path to load model weights from, e.g. `pretrained/msn/vitb16_600ep.pth.tar`
* `job_dir`: Directory to save the logs and model weights to. The parent directory (e.g. `logs/submitit/`) should be created beforehand.
* `data_path`: Specifies the path to the full ImageNet dataset, e.g. `../datasets/ImageNet/imagenet/`

Please refer to [FINETUNE.md](https://github.com/facebookresearch/mae/blob/main/FINETUNE.md) for more details on full fine-tuning on ImageNet.

## Evaluation on ImageNet and variants

The bash command used for evaluation on ImageNet and variants can be found [here](https://github.com/Aaditya-Singh/mae/blob/main/commands/Eval-MSN-IN.sh). We summarize some of the important flags for experimentation below:

* `resume`: Specifies the full path to load model weights from, e.g. `pretrained/msn/vitb16_100ep_ft.pth.tar`
* `data_path`: Specifies the path to the evaluation dataset, e.g. `../datasets/ImageNet/imagenet/`

## Bibtex
Please cite the original authors if you find this repository helpful:
```
@Article{MaskedAutoencoders2021,
  author  = {Kaiming He and Xinlei Chen and Saining Xie and Yanghao Li and Piotr Doll{\'a}r and Ross Girshick},
  journal = {arXiv:2111.06377},
  title   = {Masked Autoencoders Are Scalable Vision Learners},
  year    = {2021},
}
```
