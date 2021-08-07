# Multi-Stage Progressive Image Restoration (CVPR 2021)

[Syed Waqas Zamir](https://scholar.google.es/citations?user=WNGPkVQAAAAJ&hl=en), [Aditya Arora](https://adityac8.github.io/), [Salman Khan](https://salman-h-khan.github.io/), [Munawar Hayat](https://scholar.google.com/citations?user=Mx8MbWYAAAAJ&hl=en), [Fahad Shahbaz Khan](https://scholar.google.es/citations?user=zvaeYnUAAAAJ&hl=en), [Ming-Hsuan Yang](https://scholar.google.com/citations?user=p9-ohHsAAAAJ&hl=en), and [Ling Shao](https://scholar.google.com/citations?user=z84rLjoAAAAJ&hl=en)

**Paper**: https://arxiv.org/abs/2102.02808

**Supplementary**: [pdf](https://drive.google.com/file/d/1mbfljawUuFUQN9V5g0Rmw1UdauJdckCu/view?usp=sharing)

**Video Presentation**: https://www.youtube.com/watch?v=0SMTPiLw5Vw

**Presentation Slides**: [pdf](https://drive.google.com/file/d/1-L43wj-VTppkrR9AL6cPBJI2RJi3Hc_z/view?usp=sharing)

> **Abstract:** *Image restoration tasks demand a complex balance between spatial details and high-level contextualized information while recovering images. In this paper, we propose a novel synergistic design that can optimally balance these competing goals. Our main proposal is a multi-stage architecture, that progressively learns restoration functions for the degraded inputs, thereby breaking down the overall recovery process into more manageable steps. Specifically, our model first learns the contextualized features using encoder-decoder architectures and later combines them with a high-resolution branch that retains local information. At each stage, we introduce a novel per-pixel adaptive design that leverages in-situ supervised attention to reweight the local features. A key ingredient in such a multi-stage architecture is the information exchange between different stages. To this end, we propose a two-faceted approach where the information is not only exchanged sequentially from early to late stages, but lateral connections between feature processing blocks also exist to avoid any loss of information. The resulting tightly interlinked multi-stage architecture, named as MPRNet, delivers strong performance gains on ten datasets across a range of tasks including image deraining, deblurring, and denoising. For example, on the Rain100L, GoPro and DND datasets, we obtain PSNR gains of 4 dB, 0.81 dB and 0.21 dB, respectively, compared to the state-of-the-art.* 

## Network Architecture
<table>
  <tr>
    <td> <img src = "https://i.imgur.com/69c0pQv.png" width="500"> </td>
    <td> <img src = "https://i.imgur.com/JJAKXOi.png" width="400"> </td>
  </tr>
  <tr>
    <td><p align="center"><b>Overall Framework of MPRNet</b></p></td>
    <td><p align="center"> <b>Supervised Attention Module (SAM)</b></p></td>
  </tr>
</table>

## Installation
The model is built in PyTorch 1.1.0 and tested on Ubuntu 16.04 environment (Python3.7, CUDA9.0, cuDNN7.5).

For installing, follow these intructions
```
conda create -n pytorch1 python=3.7
conda activate pytorch1
conda install pytorch=1.1 torchvision=0.3 cudatoolkit=9.0 -c pytorch
pip install matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm
```

Install warmup scheduler

```
cd pytorch-gradual-warmup-lr; python setup.py install; cd ..
```

## Training and Evaluation

Training and Testing codes for deblurring, deraining and denoising are provided in their respective directories.

## Results
Experiments are performed for different image processing tasks including, image deblurring, image deraining and image denoising.

### Image Deblurring

<table>
  <tr>
    <td> <img src = "https://i.imgur.com/UIwmY13.png" width="450"> </td>
    <td> <img src = "https://i.imgur.com/ecSlcEo.png" width="450"> </td>
  </tr>
  <tr>
    <td><p align="center"><b>Deblurring on Synthetic Datasets.</b></p></td>
    <td><p align="center"><b>Deblurring on Real Dataset.</b></p></td>
  </tr>
</table>

### Image Deraining 

<img src = "https://i.imgur.com/YVXWRJT.png" width="900">

### Image Denoising

<p align="center"> <img src = "https://i.imgur.com/Wssu6Xu.png" width="450"> </p>

