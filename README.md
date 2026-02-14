# Dual-Path Discriminative Semantic Distillation for Self-Supervised Fine-Grained Visual Recognition

This is the  implementation of  paper: Dual-Path Discriminative Semantic Distillation for Self-Supervised Fine-Grained Visual Recognition
> ⚠ **Note:** The source code is currently incomplete and will be fully released once the manuscript is accepted by the journal.

## Datasets
Experiments on **3 image datasets**:
FGVC-Aircraft，Stanford Cars, CUB-200-2011

|#|Datasets|Download|
|---|----|-----|
|1|FGVC-Aircraft|[Link](https://www.kaggle.com/datasets/seryouxblaster764/fgvc-aircraft)|
|2|Stanford Cars|[Link](https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset)
|3|CUB-200-2011|[Link](https://www.kaggle.com/datasets/wenewone/cub2002011)  |


## Pretrained Model
You can download  pre-trained  models: [Swin-T, Pretrain on ImageNet-1k](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth)

 ## Environment
```python
pip install -r requirement.txt
```

## Training/Resume Training

```python
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29601 main_DP-DSD.py --cfg experiments/imagenet/swin/swin_tiny_patch4_window7_224.yaml
```

## Test/Evaluation

```python
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29601 eval_linear.py --arch swin_tiny --cfg experiments/imagenet/swin/swin_tiny_patch4_window7_224.yaml MODEL.NUM_CLASSES 100
```

## License & Acknowledgment

We are very grateful for these excellent works: [EsViT](https://github.com/microsoft/esvit),[DINO]( https://github.com/facebookresearch/dino). Please follow their respective licenses for usage and redistribution. Thanks for their awesome works.

## Contact

Feel free to contact me if there is any question. (Ting Yang: [yangting123@stu.ouc.edu.cn](mailto:yangting123@stu.ouc.edu.cn), Lei Huang: [huangl@ouc.edu.cn](mailto:huangl@ouc.edu.cn))

---
