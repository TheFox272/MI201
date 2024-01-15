import tensorflow as tf
import matplotlib as mpl
import os
import matplotlib.pyplot as plt
import torch, torchvision
from tensorflow.keras.optimizers import Adam
import numpy as np

W = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
net = torchvision.models.segmentation.deeplabv3_resnet50(weights=W)
net = net.cuda()

# ImageNet labels
# decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions


# Helper function to extract labels from probability vector
# def get_imagenet_label(probs):
#   return decode_predictions(probs, top=1)[0][0]

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

visu = torch.cat([im1, im2, im3, im4, im5, im6, im7, im8, im9], dim=-1)
visu = visu.cpu().numpy().transpose(1, 2, 0)
dpi = plt.rcParams['figure.dpi']
width_px = 1600
height_px = 200
# plt.figure(figsize=(width_px / dpi, height_px / dpi))
# plt.imshow(visu)
# plt.axis('off')
# plt.show()

with torch.no_grad():
  x = torch.stack([im1,im2,im3,im4,im5,im6,im7,im8,im9],dim=0)
  x = (W.transforms())(x).cuda()
  z = net(x)["out"] # on prédit des cartes de score de confiance
  z = z[:,[0,8,12,15],:,:] # we keep only person, cat and dog class
  _,z = z.max(1) # on prend le meilleur score



couleur = torch.zeros(9,3,520,520).cuda()
couleur[:,0,:,:] = (z==1).float() # red for cat
couleur[:,1,:,:] = (z==2).float() # green for dog
couleur[:,2,:,:] = (z==3).float() # blue for person
visu = torch.cat([im1,im2,im3,im4,im5,im6,im7,im8,im9],dim=-1)
visubis = torch.cat([couleur[i] for i in range(9)],dim=-1).cpu()
visu = torch.cat([visu,visubis],dim=1)
visu = visu.cpu().numpy().transpose(1,2,0)
dpi = plt.rcParams['figure.dpi']
width_px = 1600
height_px = 400
plt.figure(figsize=(width_px/dpi, height_px/dpi))
plt.imshow(visu)
plt.axis('off')
# plt.show()

def clip_eps(tensor, eps):
	# clip the values of the tensor to a given range and return it
	return tf.clip_by_value(tensor, clip_value_min=-eps,
		clip_value_max=eps)

# define the epsilon and learning rate constants
EPS = 4 / 255.0
LR = 0.1
optimizer = Adam(learning_rate=LR)

def generate_adversaries(net, baseImage, delta, steps):
	# iterate over the number of steps
    z = net(baseImage)["out"] # on prédit des cartes de score de confiance
    z = z[:,[0,8,12,15],:,:] # we keep only person, cat and dog class
    _,z = z.max(1)

    delta_final = delta.copy()
    adv = baseImage.detach().clone()
    print(adv[0,:,:,:], baseImage.cpu()[0,:,:,:], delta_final[0])   
    # print(adversary.shape())   

    for i in range (9):
        for step in range(0, steps):
            # record our gradients
            with tf.GradientTape() as tape:
                # explicitly indicate that our perturbation vector should
                # be tracked for gradient updates
                tape.watch(delta_final[i])
                # add our perturbation vector to the base image and
                # preprocess the resulting image
                adv[i,:,:,:] = (baseImage.cpu()[i,:,:,:] + delta_final[i])
                # adv[i,:,:,:] = torch.nn.functional.interpolate(adv[i,:,:,:].unsqueeze(0), size=520)[0]
                zm = net(adv)["out"] # on prédit des cartes de score de confiance
                zm = zm[:,[0,8,12,15],:,:]
                _ ,zm = zm.max(1)

                norm_loss, norm_loss_coef = -loss(z, zm)
                # check to see if we are logging the loss value, and if
                # so, display it to our terminal
                if step % 5 == 0:
                    print("step: {}, loss: {}...".format(step, norm_loss.numpy()))
            # calculate the gradients of loss with respect to the
            # perturbation vector
            gradients = tape.gradient(norm_loss, delta_final[i])
            # update the weights, clip the perturbation vector, and
            # update its value
            optimizer.apply_gradients([(gradients, delta_final[i])])
            delta_final[i].assign_add(clip_eps(delta_final[i], eps=EPS))
	# return the perturbation vector
    return delta_final

deltaList = []
baseImage = torch.stack([im1,im2,im3,im4,im5,im6,im7,im8,im9],dim=0)
baseImage = (W.transforms())(x).cuda()
delta = tf.Variable([torch.zeros_like(baseImage[i].cpu()) for i in range(9)])
# print(delta)
# generate the perturbation vector to create an adversarial example
print("[INFO] generating perturbation...")
deltaList = generate_adversaries(net, baseImage, delta, 50)
print("[INFO] creating adversarial example...")
	
im1a,im2a,im3a,im4a,im5a,im6a,im7a,im8a, aim9a = im1+deltaList[0],im2+deltaList[1],im3+deltaList[2],im4+deltaList[3],im5+deltaList[4],im6+deltaList[5],im7+deltaList[6],im8+deltaList[7],im9+deltaList[8]
with torch.no_grad():
  x = torch.stack([im1a,im2a,im3a,im4a,im5a,im6a,im7a,im8a,aim9a],dim=0)
  x = (W.transforms())(x).cuda()
  zm = net(x)["out"] # on prédit des cartes de score de confiance
  zm = zm[:,[0,8,12,15],:,:] # we keep only person, cat and dog class
  _,zm = zm.max(1) # on prend le meilleur score

couleurm = torch.zeros(9,3,520,520).cuda()
couleurm[:,0,:,:] = (zm==1).float() # red for cat
couleurm[:,1,:,:] = (zm==2).float() # green for dog
couleurm[:,2,:,:] = (zm==3).float() # blue for person   
visu = torch.cat([im1a,im2a,im3a,im4a,im5a,im6a,im7a,im8a, aim9a],dim=-1)
visubis = torch.cat([couleurm[i] for i in range(9)],dim=-1).cpu()
visu = torch.cat([visu,visubis],dim=1)
visu = visu.cpu().numpy().transpose(1,2,0)
dpi = plt.rcParams['figure.dpi']
width_px = 1600
height_px = 400
plt.figure(figsize=(width_px/dpi, height_px/dpi))
plt.imshow(visu)
plt.axis('off')
plt.show()

