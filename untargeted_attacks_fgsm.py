import matplotlib as mpl
import os
import matplotlib.pyplot as plt
import torch, torchvision
import numpy as np
import torch.nn.functional as F
from torchvision.utils import save_image

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

# net.eval()
delta = [0 for i in range(9)]
x = torch.stack([im1,im2,im3,im4,im5,im6,im7,im8,im9],dim=0)
x = (W.transforms())(x)
z_init = net(x)["out"] # on prédit des cartes de score de confiance pour les images non bruitées
z_init = z_init[:,[0,8,12,15],:,:] # we keep only person, cat and dog class
_,index = z_init.max(1)
# y = x[j][None,:,:,:]
x.requires_grad = True
for j in range (9):
    for step in range (1):
        print("step =",step)
        z = net(x)["out"] # on prédit des cartes de score de confiance pour les images non bruitées
        z = z[:,[0,8,12,15],:,:]
        print("x=",x[j])
        _,indexm = z.max(1)
        print(z[j][None,:,:,:].shape,index[j][None,:,:].shape)
        loss = F.nll_loss(z[j][None,:,:], index[j][None,:,:])
        loss.backward()            
        print("loss = ",loss)
        data_grad = x.grad 
        print("data grad", data_grad[j],data_grad.shape)
        with torch.no_grad():
            x[j] = x[j] + 4/255 * data_grad[j].sign()
            delta[j] = delta[j] + 4/255 * data_grad[j].sign()
        print("x=",x[j])
    x.grad.data.zero_()
    save_image(delta[j], 'noise_{}.png'.format(j)) 

for i in range (9) :
    print("noise norm = ", i, delta[i], torch.mean(torch.linalg.matrix_norm(delta[i])))


im = [im1,im2,im3,im4,im5,im6,im7,im8,im9]
couleurm = torch.zeros(9,3,520,520)
couleurm[:,0,:,:] = (indexm==1).float() # red for cat
couleurm[:,1,:,:] = (indexm==2).float() # green for dog
couleurm[:,2,:,:] = (indexm==3).float() # blue for person   
visum = torch.cat([im[i]  + delta[i] for i in range(9)],dim=-1).cpu()#+noise[i]
visubism = torch.cat([couleurm[i] for i in range(9)],dim=-1).cpu()

couleur = torch.zeros(9,3,520,520)
couleur[:,0,:,:] = (index==1).float() # red for cat
couleur[:,1,:,:] = (index==2).float() # green for dog
couleur[:,2,:,:] = (index==3).float() # blue for person
visu = torch.cat([im1,im2,im3,im4,im5,im6,im7,im8,im9],dim=-1)
visubis = torch.cat([couleur[i] for i in range(9)],dim=-1).cpu()

visu = torch.cat([visu,visubis,visum,visubism],dim=1)
visu = visu.cpu().numpy().transpose(1,2,0)
dpi = plt.rcParams['figure.dpi']
width_px = 1600
height_px = 400
plt.figure(figsize=(width_px/dpi, height_px/dpi))
plt.imshow(visu)
plt.axis('off')
plt.show()