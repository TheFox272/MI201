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
# img = torch.nn.functional.interpolate(im0.unsqueeze(0), size=520)[0]
batch = preprocess(im2_noise)

# Step 4: Use the model and visualize the prediction
prediction = model(batch[None,:,:,:])["out"]
normalized_masks = prediction.softmax(dim=1)
class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
mask = normalized_masks[0, class_to_idx["dog"]]
# mask2 = normalized_masks[0, class_to_idx["dog"]]
to_pil_image(mask).show()

