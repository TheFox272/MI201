import matplotlib as mpl
import os
import matplotlib.pyplot as plt
import torch, torchvision
import numpy as np


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
