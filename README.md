
##  Introduction

This repository contains the PyTorch implementation of:

Adaptive cervical cell segmentation based on local and global dependence

![ScreenShot](/images/framework.jpg)

##  Requirements

* Pytorch
* Torchvision 
* tqdm
* scipy
* skimage
* PIL
* numpy

## Datasets
* [CX22](https://github.com/LGQ330/Cx22)
#### Data Format
  * datas
    * -train
      * --images
      * --masks
    * -valid
      * --images
      * --masks
    * -test
      * --images
      * --masks

##  Usage

####  1. Training

```bash
python train.py  --mode train  --dataset CX22  
--train_data_dir /path-to-train_data  --valid_data_dir  /path-to-valid_data
```



####  2. Inference

```bash
python test.py  --mode test  --load_ckpt checkpoint 
--dataset CX22    --test_data_dir  /path-to-test_data
```



##  Citation

It will be provided soon:

@ARTICLE{
}


## References 

* [Relation Networks](https://github.com/milesial/Pytorch-UNet)

