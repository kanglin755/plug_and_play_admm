# ADMM Plug and Play
This repository provides a full Pytorch example for ADMM plug and play in the context of image restoration. <br>
The example ranges from downloading training data and training a CNN denoiser to plugging in the denoiser into ADMM plug and play for a motion deblur and inpainting example.  <br>
[View in Google Colab](https://colab.research.google.com/drive/1XHGdKA-eTvOUwto2jA89z1SEfLTdR-jN?usp=sharing)

|| Degraded | PnP output | Ground truth | 
|-- |--|--|--| 
|Motion deblur|![](figs/degraded_motion.png) | ![](figs/pnp_motion.png) | ![](figs/image_motion.png) |
|Inpainting|![](figs/degraded_inpainting.png) | ![](figs/pnp_inpainting.png) | ![](figs/image_inpainting.png) |
|Super-resolution|![](figs/degraded_superres.png) | ![](figs/pnp_superres.png) | ![](figs/image_superres.png) |
