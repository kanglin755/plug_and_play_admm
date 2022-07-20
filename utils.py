import torch
import torch.nn as nn

def conv2d_from_kernel(kernel, channels, device, stride=1):
    """
    Returns nn.Conv2d and nn.ConvTranspose2d modules from 2D kernel, such that 
    nn.ConvTranspose2d is the adjoint operator of nn.Conv2d
    Arg:
        kernel: 2D kernel
        channels: number of image channels
    """
    kernel = kernel/kernel.sum()
    kernel = kernel.repeat(channels, 1, 1, 1)

    filter = nn.Conv2d(
        in_channels=channels, out_channels=channels,
        kernel_size=kernel_size, groups=channels, bias=False, stride=stride,
        # padding=kernel_size//2
    )
    filter.weight.data = kernel
    filter.weight.requires_grad = False

    filter_adjoint = nn.ConvTranspose2d(
        in_channels=channels, out_channels=channels,
        kernel_size=kernel_size, groups=channels, bias=False, stride=stride,
        # padding=kernel_size//2,
    )
    filter_adjoint.weight.data = kernel
    filter_adjoint.weight.requires_grad = False

    return filter.to(device), filter_adjoint.to(device)

def compute_psnr(img1, img2):
    mse = torch.mean((img1*255 - img2*255) ** 2)
    return 20 * torch.log10(255.0 / torch.sqrt(mse))
