# import tensorflow as tf
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


def clip_eps(tensor, eps):
    return torch.clip(tensor, eps,-eps)  

def loss(im1,im2):
    s=[]
    s.append(torch.mean(torch.linalg.matrix_norm((im1-im2).double())))
    return torch.stack(s,dim=0)

#Fast gradient sign method for untargeted attack
def generate_adversaries(net,baseImage, adv):
    delta = []
    adv = (W.transforms())(adv)
    #pour chaque image, on évalue la prédiction, dont on calcule la loss, puis on calcule le gradient de la loss par rapport à l'image et on update adv
    for j in range (9):
        net.eval()
        a = adv[j][None,:,:,:]
        a.requires_grad = True  
        zm = net(a)["out"] # on prédit des cartes de score de confiance
        _ ,zm = zm.max(1)

        norm_loss = torch.negative(loss(zf, zm))
        print("loss = ",norm_loss)

        norm_loss.requires_grad = True
        norm_loss.mean().backward(retain_graph=True)
        data_grad = norm_loss.grad
        print(data_grad)
        print("data_grad = ",data_grad.sign())
        print("delta = ", EPS*data_grad.sign())
        print("adv = ", j,adv[j].shape, baseImage[j].shape, EPS*data_grad.sign())
        adv[j] = baseImage[j] + EPS*data_grad.sign()
        delta.append(EPS*data_grad.sign())
        norm_loss.grad.zero_()

    return delta

# generate the perturbation vector to create an adversarial example
print("[INFO] generating perturbation...")
print("[INFO] creating adversarial example...")

# define the epsilon
EPS = 0.1
x = torch.stack([im1,im2,im3,im4,im5,im6,im7,im8,im9],dim=0)
baseImage = (W.transforms())(x)
zf = net(baseImage)["out"] # on prédit des cartes de score de confiance pour le calcul de la loss dans generate_adversaries
_,zf = zf.max(1)
noise = torch.randn_like(baseImage)*EPS
adv = baseImage + noise #on inititialise les adversaires avec du bruit
delta = generate_adversaries(net, baseImage, adv) # delta est la perturbation à appliquer à chaque image

# Visualisation d'une image avec bruit
# print(noise.shape,noise)
# plt.imshow((im1 + noise[0]+delta[0]).numpy().transpose(1,2,0))
# plt.show()

net.train()	

## Visualisation
with torch.no_grad():
  adversaries = (W.transforms())(adv)
  zm = net(adversaries)["out"] # on prédit des cartes de score de confiance pour les adversaires
  zm = zm[:,[0,8,12,15],:,:] # we keep only person, cat and dog class
  _,zm = zm.max(1) # on prend le meilleur score

  x = torch.stack([im1,im2,im3,im4,im5,im6,im7,im8,im9],dim=0)
  x = (W.transforms())(x)
  z = net(x)["out"] # on prédit des cartes de score de confiance pour les images non bruitées
  z = z[:,[0,8,12,15],:,:] # we keep only person, cat and dog class
  _,z = z.max(1) # on prend le meilleur score

couleur = torch.zeros(9,3,520,520)
couleur[:,0,:,:] = (z==1).float() # red for cat
couleur[:,1,:,:] = (z==2).float() # green for dog
couleur[:,2,:,:] = (z==3).float() # blue for person
visu = torch.cat([im1,im2,im3,im4,im5,im6,im7,im8,im9],dim=-1)
visubis = torch.cat([couleur[i] for i in range(9)],dim=-1).cpu()

im = [im1,im2,im3,im4,im5,im6,im7,im8,im9]
couleurm = torch.zeros(9,3,520,520)
couleurm[:,0,:,:] = (zm==1).float() # red for cat
couleurm[:,1,:,:] = (zm==2).float() # green for dog
couleurm[:,2,:,:] = (zm==3).float() # blue for person   
visum = torch.cat([im[i] +noise[i] + delta[i] for i in range(9)],dim=-1).cpu()
visubism = torch.cat([couleurm[i] for i in range(9)],dim=-1).cpu()

visu = torch.cat([visu,visubis,visum,visubism],dim=1)
visu = visu.cpu().numpy().transpose(1,2,0)
dpi = plt.rcParams['figure.dpi']
width_px = 1600
height_px = 400
plt.figure(figsize=(width_px/dpi, height_px/dpi))
plt.imshow(visu)
plt.axis('off')
plt.show()
