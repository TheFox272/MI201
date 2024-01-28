import matplotlib as mpl
import os
import matplotlib.pyplot as plt
import torch, torchvision
import numpy as np
import torch.nn.functional as F

from torchvision.io.image import read_image
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms.functional import to_pil_image


W = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
net = torchvision.models.segmentation.deeplabv3_resnet50(weights=W)
# net = net.cuda()

os.chdir("./coco")

im1 = torchvision.io.read_image("217730183_8f58409e7c_z.jpg").float() / 255
im2 = torchvision.io.read_image("541870527_8fe599ec04_z.jpg").float() / 255
im3 = torchvision.io.read_image("2124681469_7ee4868747_z.jpg", mode=torchvision.io.ImageReadMode.RGB).float() / 255
im4 = torchvision.io.read_image("2711568708_89f2308b85_z.jpg").float() / 255
im5 = torchvision.io.read_image("2928196999_acd5471d23_z.jpg").float() / 255
im6 = torchvision.io.read_image("3016145160_497da1b387_z.jpg").float() / 255
im7 = torchvision.io.read_image("4683642953_2eeda0820e_z.jpg").float() / 255
im8 = torchvision.io.read_image("6911037487_cc68a9d5a4_z.jpg").float() / 255
im9 = torchvision.io.read_image("8139728801_60c233660e_z.jpg").float() / 255
im1 = torch.nn.functional.interpolate(im1.unsqueeze(0), size=520)[0]
im2 = torch.nn.functional.interpolate(im2.unsqueeze(0), size=520)[0]
im3 = torch.nn.functional.interpolate(im3.unsqueeze(0), size=520)[0]
im4 = torch.nn.functional.interpolate(im4.unsqueeze(0), size=520)[0]
im5 = torch.nn.functional.interpolate(im5.unsqueeze(0), size=520)[0]
im6 = torch.nn.functional.interpolate(im6.unsqueeze(0), size=520)[0]
im7 = torch.nn.functional.interpolate(im7.unsqueeze(0), size=520)[0]
im8 = torch.nn.functional.interpolate(im8.unsqueeze(0), size=520)[0]
im9 = torch.nn.functional.interpolate(im9.unsqueeze(0), size=520)[0]

im0_noise = torchvision.io.read_image("im0_noise.png").float()/255
im1_noise = torchvision.io.read_image("im1_noise.png").float()/255
im2_noise = torchvision.io.read_image("im2_noise.png").float()/255
im3_noise = torchvision.io.read_image("im3_noise.png").float()/255
im4_noise = torchvision.io.read_image("im4_noise.png").float()/255
im5_noise = torchvision.io.read_image("im5_noise.png").float()/255
im6_noise = torchvision.io.read_image("im6_noise.png").float()/255
im7_noise = torchvision.io.read_image("im7_noise.png").float()/255
im8_noise = torchvision.io.read_image("im8_noise.png").float()/255


# Step 1: Initialize model with the best available weights
weights = FCN_ResNet50_Weights.DEFAULT
model = fcn_resnet50(weights=weights)
# model.cuda()
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
y = torch.stack([im0_noise,im1_noise,im2_noise,im3_noise,im4_noise,im5_noise,im6_noise,im7_noise,im8_noise],dim=0)
x = torch.stack([im1,im2,im3,im4,im5,im6,im7,im8,im9],dim=0)
batch_x = preprocess(x)
batch_y = preprocess(y)
# Step 4: Use the model and visualize the prediction
prediction_x = model(batch_x)["out"]
prediction_y = model(batch_y)["out"]
normalized_masks_x = prediction_x.softmax(dim=1)
normalized_masks_y = prediction_y.softmax(dim=1)

class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
visu_x = []
visu_y = []
for i in range (9):
    im_x = normalized_masks_x[i, class_to_idx["dog"]].detach()
    visu_x.append(torch.nn.functional.interpolate(im_x.unsqueeze(0), size=520)[0])
    im_y = normalized_masks_y[i, class_to_idx["dog"]].detach()
    visu_y.append(torch.nn.functional.interpolate(im_y.unsqueeze(0), size=520)[0])

visu_x = torch.cat(visu_x, dim=-1)
# print((visu_x[0]).shape)
visu_y = torch.cat(visu_y, dim=-1)
visu = torch.cat([visu_x,visu_y],dim=0)
dpi = plt.rcParams['figure.dpi']
width_px = 1600
height_px = 400
plt.figure(figsize=(width_px/dpi, height_px/dpi))
plt.imshow(visu)
plt.axis('off')
plt.show()

