# import tensorflow as tf
# from tarfile import data_filter
import matplotlib as mpl
import os
import matplotlib.pyplot as plt
import torch, torchvision
import numpy as np
import torch.nn.functional as F


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

def retrieve_grad(grad,index,new_grad):
    #grad = tensor de la forme nb_image,nb_classes,hauteur,largeur
    #ici de la forme 1,4,520,520
    #index de la forme 1,520,520
    # print(new_grad.shape)
    for i in range (len(index[0])):
        for j in range (len(index[0][0])):
            ind = index[0][i][j]
            new_grad[i][j] += grad[0][ind][i][j]
    # print(new_grad)
    return new_grad

#Fast gradient sign method for untargeted attack
def generate_adversaries(net,baseImage, adv, noise, target):
    delta = [0 for i in range(9)]
    nb_classes = 4
    batch_size, height, width = 1,520,520   
    adv = (W.transforms())(adv)
    #pour chaque adv, on évalue la prédiction, dont on calcule la loss, 
    # puis on calcule la loss de la target (= 1,2,3 pour cat, dog, person)
    # puis on calcule la loss totale, qui correspond à la loss de la target moins la loss de l'image
    # en effet, on cherche à maximiser la loss de la target et minimiser la loss de l'image
    #puis on calcule le gradient de la loss et on update adv
    for j in range (9):
        net.eval()
        a = adv[j][None,:,:,:]
        # print("a = ",a)
        a.requires_grad = True  
        new_grad=torch.zeros([520, 520])
        for k in range (2):
            print("step = ",k)
            print("a = ",a,a.shape)
            zm = net(a)["out"] # on prédit des cartes de score de confiance
            print("zm = ",zm.shape)
            with torch.no_grad():
                zm = zm[:,[0,8,12,15],:,:] # we keep only person, cat and dog class

                # print("zm = ",zm,zm.shape)
                # zm = zm.amax(1) pour autre loss 
            # print("zm = ",zm,zm.shape)
            print("index = ",index[j][None,:,:],index[j][None,:,:].shape)
            print(zm.shape, index[j][None,:,:].shape)
            # print("zm = ",zm)
            zm.requires_grad = True  
            loss = F.nll_loss(zm, index[j][None,:,:])
            # a.retain_grad = True
            print("a =",a)
            loss.backward()#retain_graph=True)
            # print("backward ok")
            # norm_loss = torch.negative(loss(zf[j], zm))
            # norm_loss.requires_grad = True
            # norm_loss.mean().backward(retain_graph=True)
            print("loss = ",loss)
            #zm,index = zm.max(1) #on prend le meilleur score
            data_grad = a.grad #can't get input grad
            print("data grad", data_grad,data_grad.shape)
            zm,indexm = zm.max(1) #on prend le meilleur score

            # fontion recup grad : if index[0][i][j] = ind (dans 0,1,2,3), alors on garde new_grad.append(grad[0][ind]) de la forme 520*520 puis new_grad[None,:,:]
            data_grad = retrieve_grad(data_grad,indexm,new_grad)
            # print("test",data_grad[0][0])
            # data_grad= data_grad.amin(1)
            print("grad index",data_grad.shape, data_grad) 
            
            #check zm max
            # chose grad, add eps*sign(grad) to adv IMAGE not zm
            # data_grad[] = data_grad[indexm[]]
            # print("data_grad = ",data_grad.sign())
            print("delta = ", EPS*data_grad.sign())
            print("adv = ", j,(EPS*data_grad.sign()).shape, EPS*data_grad.sign())
            a = (a + EPS*data_grad.sign())
            print("a = ",a.shape)
            # print("argmax",zm.argmax(1))
            delta[j] += (EPS*data_grad.sign())
            # print("delta = ",delta[j].shape,delta[j])
        # zm.grad.zero_()
        with torch.no_grad():
            # print("zm amax", zm.amax(1))
            adv[j] = a
    return delta

# generate the perturbation vector to create an adversarial example
print("[INFO] generating perturbation...")
print("[INFO] creating adversarial example...")

with torch.no_grad():
# define the epsilon
    net.train()	
    EPS = 0.1
    x = torch.stack([im1,im2,im3,im4,im5,im6,im7,im8,im9],dim=0)
    baseImage = (W.transforms())(x)
    zf = net(baseImage)["out"] # on prédit des cartes de score de confiance pour le calcul de la loss dans generate_adversaries
    zf = zf[:,[0,8,12,15],:,:] # we keep only background, person, cat and dog class
    # print("zf 1", zf[1], zf[0].shape)
    zf,index = zf.max(1) 
    # print("zf",zf[1], zf[0].shape)
    # print("indice",index, index.shape)

    noise = torch.randn_like(baseImage)*EPS
    adv = baseImage #+ noise #on inititialise les adversaires avec du bruit
    # t = torch.randint(0, 4, (1, 520,520))
    # print(t, t.shape)
    print("base,adv", baseImage.shape,adv.shape)
delta = generate_adversaries(net, baseImage, adv, noise,1) # delta est la perturbation à appliquer à chaque image

# Visualisation d'une image avec bruit
# print(noise.shape,noise)
# plt.imshow((noise[0]).numpy().transpose(1,2,0))#+delta[0]
# print(delta[0].shape,delta.shape)
# plt.imshow((delta[0]).numpy().transpose(1,2,0))#+

plt.show()

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
#   print("z final",z)

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
visum = torch.cat([im[i]  + delta[i] for i in range(9)],dim=-1).cpu()#+noise[i]
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
