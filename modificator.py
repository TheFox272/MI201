import torch

max_noise = 4/255

def gaussian_noise(image):
    noise = torch.randn_like(image)
    noise *= max_noise / torch.mean(torch.linalg.matrix_norm(noise))
    return image + noise
















