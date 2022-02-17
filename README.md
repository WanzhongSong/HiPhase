# HiPhase
This code implements the approach as
described in the following research paper:

 * Deep absolute phase recovery from single-frequency phase map for handheld 3D measurement  
 * Songlin Bai, Xiaolong Luo, Kun Xiao, Chunqian Tan and Wanzhong Song*  
 * **Optics Communications**, 2022(512) [[PDF](https://authors.elsevier.com/a/1ebea6wPvpGNF)] 


## Highlights
 * Absolute fringe-order is retrieved from one FPP phase map by the DCNN  
 * The DCNN is lightweight and operates in real-time for a phase-map of 1024×1024 pixels on a GTX 1660Ti.  
 * A large-scale and challenging phase unwrapping dataset is built from real objects and publicly available.


## Preamble
This code was developed and tested with python 3.6, Pytorch 1.8.0, and CUDA 10.2 on Ubuntu 18.04. It is based on [Eduardo Romera's ERFNet implementation (PyTorch Version)](https://github.com/Eromera/erfnet_pytorch). 


## Prerequisite
install manually the following packages :

```
torch
PIL
numpy
argparse
```


## Datasets
Our raw data SCU-Phase-RawData will be available.

Our ready dataset is [SCU-Phase-ReadyData](https://pan.baidu.com/s/Xa7P0ZGWeO3oLVCIgvhLWg code: h3bc).  



## Training
Training the HiPhase model from scratch on SCU-Phase-ReadyData by running
```bash
python train/main.py
```


## Evaluation

Evaluating the trained model by running
```bash
python eval/eval_gray.py
```
Evaluating the mIoU by running
```bash
python eval/eval_iou.py
```


## Pretrained Model

Our pretrained HiPhase model is [HiPhase-experi](https://github.com/WanzhongSong/HiPhase/blob/main/model_best.pth)


    
## Citation
```
@article{Bai2022,
  author = {Bai, Songlin and Luo, Xiaolong and Xiao, Kun and Tan, Chunqian and Song, Wanzhong},  
  title = {Deep absolute phase recovery from single-frequency phase map for handheld 3D measurement},
  journal = {Optics Communications},
  publisher = {Elsevier Ltd.},
  volume = {512},
  year = {2022}
}
```

## License
This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License, which allows for personal and research use only. For a commercial license please contact the authors. You can view a license summary here: http://creativecommons.org/licenses/by-nc/4.0/