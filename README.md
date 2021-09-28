# Deep SR-HDR
## Deep SR-HDR: Joint Learning of Super-Resolution and High Dynamic Range Imaging for Dynamic Scenes
By Xiao Tan, Huaian Chen, Kai Xu, Yi Jin, Changan Zhu

### Highlights
- **a novel scheme and a novel efficient network**: We propose an efficient scheme and a novel efficient network for the joint SR-HDR task. To the best of our knowledge, this is the first CNN-based method for multi-fame SR-HDR imaging of dynamic scenes.
- **a multi-scale deformable module (MSDM)**: To handle large complex motions, we devise an MSDM that estimates the sampling location offsets in a coarse-to-fine manner, helping flexibly integrate useful information to compensate for the occluded content in the motion regions instead of explicitly aligning these regions.
- **State of the art**

## Dependencies and Installation

- Python 3.6.9 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.1](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- [Deformable Convolution](https://arxiv.org/abs/1703.06211). We use [mmdetection](https://github.com/open-mmlab/mmdetection)'s dcn implementation. Please first compile it.
  ```
  cd ./dcn
  python setup.py develop
  ```

## Dataset Preparation Using MATLAB
We use datasets in h5 format for faster IO speed. 
Please unzip the [training and test datasets](https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/) into ./dataset_select/Data

Note that put all test scenes in a folder 'Test'.
  ```
  cd ./dataset_full
  run PrepareData
  ```

## Train
Note that x2 model need to load x4 model as initialization for better convergence.
  ```
  python train.py
  ```

## Test SR-HDR
  ```
  python test.py
  ```

## Test HDR (full resolution)
  ```
  cd HDR_full_resolution
  python test.py
  ```
  
