# caption2image

PyTorch implementation of [GAN-INT-CLS](http://arxiv.org/abs/1605.05396) and [AttnGAN](http://openaccess.thecvf.com/content_cvpr_2018/html/Xu_AttnGAN_Fine-Grained_Text_CVPR_2018_paper.html)

## Dependencies
- python 3
- Pytorch 1.0.0
- tensorflow, tensorboard (you can train/evaluate your model without this if you do not use tensorboard for logger)

In addition, you may need other packages...

## Data

1. Download preprocessed metadata for [COCO filename](https://drive.google.com/file/d/1cYcHZNJ63tHjMITvBdf8pWD7FNAWXxnq/view?usp=sharing) and [COCO text](https://drive.google.com/file/d/1sz3kuQfmGKirv7pIaqrtr7SF0ZdzMPT4/view?usp=sharing) and extract
2. Download [COCO dataset](http://cocodataset.org/#download)
3. Place data as below

```
data_dir 
  |- COCO
       |- filenames 
            |- train2014 
            |- val2014 
       |- text 
            |- train2014 
            |- val2014 
       |- image 
            |- train2014 
            |- val2014 
```

## AttnGAN

### Training

- Train DAMSM models:
  - `python DAMSM_main.py`
    - you can edit config by directly editting the source code
 
- Train AttnGAN models:
  - `python main.py`
    - you can edit config by passing arguments (see AttnGAN/config.py or `python main.py --help`)

### Evaluation

I prepared notebook for evaluation (AttnGAN/eval.ipynb).  
You can evaluate generated images by
- inception score
- frechet inception distance
- R-precision  

You can also generate images from your own captions.

### Pretrained Model

1. Download [DAMSM image_encoder](https://drive.google.com/file/d/1OJE_g2Oh-VIVEt0ulVmHnc8hSn3m22q1/view?usp=sharing)
2. Download [DAMSM text_encoder](https://drive.google.com/file/d/1DKSbE5kSoGnBoiJQFhWFl-oeBe7CXiTh/view?usp=sharing)
3. Download [AttnGAN Generator](https://drive.google.com/file/d/1XysSOSmGhLJ-ZoTr0AY_DOTYOS_Ztm2J/view?usp=sharing) and [config](https://drive.google.com/file/d/1r6qQqCiTlIo93tPyZ84_fmB6ZWhKGBqZ/view?usp=sharing)
4. Place models as below

```
AttnGAN
  |- results
       |- DAMSM/COCO/2019_05_04_00_32/model
            |- image_encoder600.pth
            |- text_encoder600.pth
       |- AttnGAN/COCO/2019_05_14_17_08
            |- model
                 |- G_epoch50.pth
            |- config.txt
```
## TODO
- Paper
  - [x] GAN-INT-CLS
    - [x] survey
    - [x] impl
  - [ ] StackGAN
    - [x] survey
    - [ ] impl
  - [ ] StackGAN++
    - [x] survey
    - [ ] impl
  - [x] AttnGAN
    - [x] survey
    - [x] impl
  - [ ] MirrorGAN
    - [ ] survey
    - [ ] impl
- Dataset
  - [ ] Bird
  - [x] MS COCO

## Reference 
- [GAN-INT-CLS](http://arxiv.org/abs/1605.05396) ([code](https://github.com/reedscot/icml2016))
- [AttnGAN](http://openaccess.thecvf.com/content_cvpr_2018/html/Xu_AttnGAN_Fine-Grained_Text_CVPR_2018_paper.html) ([code](https://github.com/taoxugit/AttnGAN))
