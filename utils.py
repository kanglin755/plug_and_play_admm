import torch
import torch.nn as nn
from torch.utils.data import Dataset
import PIL.Image as Image
import torchvision.transforms as pth_transforms
import matplotlib.pyplot as plt

def conv2d_from_kernel(kernel, channels, device, stride=1):
    """
    Returns nn.Conv2d and nn.ConvTranspose2d modules from 2D kernel, such that 
    nn.ConvTranspose2d is the adjoint operator of nn.Conv2d
    Arg:
        kernel: 2D kernel
        channels: number of image channels
    """
    kernel_size = kernel.shape
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

def myplot(degraded, reconstruction, target):
    plt.figure(figsize=(10,5))
    plt.subplot(1,3,1); plt.imshow(degraded.permute(0,2,3,1).squeeze().cpu()); plt.title('Degraded')
    plt.subplot(1,3,2); plt.imshow(reconstruction.permute(0,2,3,1).squeeze().cpu()); plt.title('Reconstruction')
    plt.subplot(1,3,3); plt.imshow(target.permute(0,2,3,1).squeeze().cpu()); plt.title('Ground truth')
    plt.show()
    
class ImagenetDataset(Dataset):
    def __init__(self, img_files, is_train=True):
        self.files = img_files
        self.is_train = is_train
        self.train_transform = pth_transforms.Compose([      
            pth_transforms.Resize(480),
            pth_transforms.GaussianBlur(kernel_size=3, sigma=1),
            pth_transforms.RandomCrop(128),
            pth_transforms.ToTensor(),             
            ])

        self.test_transform = pth_transforms.Compose([      
            pth_transforms.Resize(480),
            pth_transforms.GaussianBlur(kernel_size=3, sigma=1),
            pth_transforms.CenterCrop(128),
            pth_transforms.ToTensor(),             
            ])

    def __len__(self, ):
        
        return len(self.files)
    
    def __getitem__(self, idx):
        image = Image.open(self.files[idx]).convert("RGB")
        
        if self.is_train:
            image = self.train_transform(image)
        else:
            image = self.test_transform(image)
        
        sample = dict()
        noise = torch.rand(1)*0.2
        sample['noisy'] = image + noise*torch.randn_like(image)
        sample['target'] = image

        return sample
