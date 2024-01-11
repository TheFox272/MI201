import torch


def gaussian_noise(image, noise_std=0.1):
    noise = torch.randn_like(image) * noise_std
    return image + noise
















