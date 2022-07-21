# Plug and Play ADMM
This repository provides a simple Pytorch implementation of plug and play ADMM with examples. <br> 

The notebook `pnp_admm_example.ipynb` contains a example in which a pretrained convnet gaussian denoiser is downloaded (120MB) and plugged into ADMM for solving a motion deblur, inpainting, and super-resolution problem. You can also [view in Google Colab](https://colab.research.google.com/drive/1XHGdKA-eTvOUwto2jA89z1SEfLTdR-jN?usp=sharing). <br>

The notebook `denoiser_training.ipynb` contains code for training a denoiser from scratch using a subset of ImageNet as trainingset. The trainingset will be download automatically (250MB). You can also [view in Google Colab](https://colab.research.google.com/drive/1XHGdKA-eTvOUwto2jA89z1SEfLTdR-jN?usp=sharing).

|| Degraded | PnP output | Ground truth | 
|-- |--|--|--| 
|Motion deblur|![](figs/degraded_motion.png) | ![](figs/pnp_motion.png) | ![](figs/image_motion.png) |
|Inpainting|![](figs/degraded_inpainting.png) | ![](figs/pnp_inpainting.png) | ![](figs/image_inpainting.png) |
|Super-resolution|![](figs/degraded_superres.png) | ![](figs/pnp_superres.png) | ![](figs/image_superres.png) |
