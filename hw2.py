import os
import random

import numpy as np
import torch
from PIL import Image
from PIL import ImageFilter
from PIL import ImageFont
from PIL import ImageDraw
from PIL import ImageChops
from PIL import ImageTk
from torchvision import transforms
import matplotlib.pyplot as plt

os.getcwd()

# Task 2: convert the images into neural-network-compatible tensors
# using the functionality provided by Torchvision.
im1 = Image.open("IMG0.jpeg")
im2 = Image.open("IMG1.jpeg")
print(im1.format, im1.size, im1.mode,im2.format, im2.size, im2.mode)
W,H = im1.size  # The sizes of photos are same.
images = torch.zeros((2,3,H,W)).float()
print(images)
print(images.size())
xform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])

num_images = 2
for i in range(num_images):
    imagefile_name = "IMG%s.jpeg" % i
    temp = Image.open(imagefile_name)
    images[i] = xform(temp)

print(images[0])

# Task 3: Calculate the histograms for the R, G, and B channels of the images
# in their tensor representation.
def hist_gen(images):
    num_bins = 10
    hist_Tensor_Images = torch.zeros(images.shape[0],3,num_bins,dtype=torch.float)
    print(images.shape)
    for idx_in_batch in range(images.shape[0]):
        color_channels = [images[idx_in_batch,ch] for ch in range(3)]
        hists = [torch.histc(color_channels[ch],bins=num_bins,min=-3.0,max=3.0) for ch in range(3)]
        hists = [hists[ch].div(hists[ch].sum()) for ch in range(3)]
        for ch in range(3):
            hist_Tensor_Images[idx_in_batch,ch] = hists[ch]
    print(hist_Tensor_Images.shape)
    return hist_Tensor_Images
hist_Tensor_Images = hist_gen(images)

def plot_hist(hist_Tensor_Images):
    num_bins = 10
    color = ['Red','Green','Blue']
    for idx_in_batch in range(images.shape[0]):
        hist_Tensor_idx = hist_Tensor_Images[idx_in_batch]
        for ch in range(3):
            plt.subplot(num_images,3,ch+idx_in_batch*3+1)
            plt.hist(hist_Tensor_idx[ch],bins=num_bins,range=(0.0,1.0))
            plt.xlabel(color[ch])

    plt.suptitle('Comparison between two different viewpoints')
    plt.show()
plot_hist(hist_Tensor_Images)

# Task 4: Calculate the distance between the histograms on a per-channel basis
# using the Wassertein Distance
from scipy.stats import wasserstein_distance

def dist_gen(hist_Tensor):
    for ch in range(3):
        print("\n image index in channel = %d:" % ch)
        hist_Tensor_ch = torch.squeeze(hist_Tensor_Images[:,ch])

        dist = wasserstein_distance(torch.squeeze(hist_Tensor_ch[0]).cpu().numpy(),torch.squeeze(hist_Tensor_ch[1]).cpu().numpy())
        print("\n Wasserstein distance between batch: ", dist)
dist_gen(hist_Tensor_Images)

# Task 5: Select affine transformation and apply it to the two images and see what effect that has on the histograms and their distances.
torch.manual_seed(295)
affine_transformer = transforms.RandomAffine(degrees=(30,70), translate=(0.1, 0.3), scale=(0.5, 0.75))

affine_images = [affine_transformer(images[idx_in_batch]) for idx_in_batch in range(images.shape[0])]
affine_images = torch.stack((affine_images))

print("Affine transformation")
print(affine_images .shape)
plot_hist(hist_gen(affine_images))
dist_gen(affine_images)

# Task 6: Select perspective transformation and apply it to the two images and see what effect that has on the histograms and their distances.
perspective_transformer = transforms.RandomPerspective(distortion_scale=0.6, p=1.0)
perspective_images = [perspective_transformer(images[idx_in_batch]) for idx_in_batch in range(images.shape[0])]
perspective_images = torch.stack(perspective_images)
print("Perspective transformation")
print(perspective_images .shape)
plot_hist(hist_gen(perspective_images))
dist_gen(perspective_images)