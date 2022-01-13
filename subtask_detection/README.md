# Code for Subtask Segment Detection

## Introduction

The code for this section is built upon the [Compact Generalized Neural Network implementation](https://github.com/KaiyuYue/cgnl-network.pytorch). We use the CGNL and NL models trained on [ImageNet](http://image-net.org/index). We augment these models with gaze data along with corresponding egocentric image frames. The changes made are in the files `train_val.py` and `models/resnet.py`.

## Requirements

  * PyTorch >= 0.4.1 or 1.0 from a nightly release
  * Python >= 3.5
  * torchvision >= 0.2.1
  * termcolor >= 1.1.0

## Environment

The code is tested under 8 Titan-V GPUS cards on Linux with installed CUDA-9.2/8.0 and cuDNN-7.1.

## Getting Started

### Prepare Dataset

  - Download pytorch imagenet pretrained model for ResNet-50 architecture from [pytorch model zoo](https://pytorch.org/docs/stable/model_zoo.html#module-torch.utils.model_zoo). The optional download links can be found in [torchvision](https://github.com/pytorch/vision/tree/master/torchvision/models). Put them in the `pretrained` folder. A pretrained CGNL model for ImageNet is also available [here](https://drive.google.com/file/d/1ezE6_tblZdoFZTYw24NJIaP_A5E0xduS/view?usp=sharing).
  - Download our [trained models](https://drive.google.com/drive/folders/11O_MrISibL89D7bJKWEm3E_ijl7vQJun?usp=sharing) on 18 users for video demonstrations and place them in the `trained_models` folder and the corresponding [data](https://drive.google.com/open?id=1P_xzCHrvOgGdbh49_PRrBYWofJ7rpZ-W) for validation is in the `data` folder. This is a test dataset only for purposes of this review.

### To Perform Validation, we ran:
- NL network using gaze for video demos
```bash
$ python train_val.py --arch '50' --dataset 'lfd-v' --nl-type 'nl' --nl-num 1 --warmup  --val 'dummy.list' --checkpoints trained_models/gaze-model-best.pth.tar  --valid --gaze --debug
```

- Basic NL network for video demos
```bash
$ python train_val.py --arch '50' --dataset 'lfd-v' --nl-type 'nl' --nl-num 1 --warmup  --val 'dummy.list' --checkpoints trained_models/nogaze-model-best.pth.tar  --valid --debug
```

### Training Nl or CGNL networks (training data cannot be released under IRB restrictions)

```bash
$ python train_val.py --arch '50' --dataset 'lfd-v' --nl-type 'nl' --nl-num 1 --warmup --train 'train.list' --gaze
```

## License

The original code was released under the MIT License. See [LICENSE](LICENSE) for additional details.
