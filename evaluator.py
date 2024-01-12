import os
import torch
import torchvision
import matplotlib.pyplot as plt
from modificator import gaussian_noise

def load_images():
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

    return [im1, im2, im3, im4, im5, im6, im7, im8, im9]

def visualize_and_evaluate_noise(image_noise_function):
    images = load_images()
    x = torch.stack(images, dim=0)

    W = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    net = torchvision.models.segmentation.deeplabv3_resnet50(weights=W)

    # with torch.no_grad():
    #     x_transformed = (W.transforms())(x)
    #     z_original = net(x_transformed)["out"][:, [0, 8, 12, 15], :, :]
    #     _, z_original = z_original.max(1)
    # torch.save(z_original, "../data/z_original.t")

    z_original = torch.load("../data/z_original.t")

    images_noisy = [image_noise_function(image) for image in images]
    x_noisy = torch.stack(images_noisy, dim=0)

    with torch.no_grad():
        x_noisy_transformed = (W.transforms())(x_noisy)
        z_noisy = net(x_noisy_transformed)["out"][:, [0, 8, 12, 15], :, :]
        _, z_noisy = z_noisy.max(1)

    step = 3
    avg_score = 0
    for i in range(0, len(images), step):
        fig, axs = plt.subplots(step, 6, figsize=(20, 20))

        for j in range(step):
            original_image = images[i + j].cpu().numpy().transpose(1, 2, 0)
            original_prediction = z_original[i + j].cpu().numpy()
            noise = x_noisy[i + j] - x[i + j]
            noisy_image = images_noisy[i + j].cpu().numpy().transpose(1, 2, 0)
            noisy_prediction = z_noisy[i + j].cpu().numpy()

            noise_norm = torch.mean(torch.linalg.matrix_norm(noise))
            changed_pred_norm = torch.linalg.matrix_norm((z_original[i + j] - z_noisy[i + j]).double())

            axs[j, 0].imshow(original_image)
            axs[j, 0].set_title(f"Original {i + j + 1}")

            axs[j, 1].imshow(original_prediction)
            axs[j, 1].set_title(f"Prediction {i + j + 1}")

            axs[j, 2].imshow(noise.cpu().numpy().transpose(1, 2, 0))
            axs[j, 2].set_title(f"Noise {i + j + 1}\nNorm2 : {noise_norm.item():.4f}")

            axs[j, 3].imshow(noisy_image)
            axs[j, 3].set_title(f"Noisy {i + j + 1}")

            axs[j, 4].imshow(noisy_prediction)
            axs[j, 4].set_title(f"Noisy Prediction {i + j + 1}")

            actual_score = score(noise_norm.item(), changed_pred_norm.item())
            avg_score += actual_score

            axs[j, 5].axis('off')
            axs[j, 5].text(0, 0.5, f"Score = {actual_score}", fontsize=10)

        plt.show()

    return avg_score / 9

def score(noise_norm, changed_pred_norm, noise_factor=2000):
    print(changed_pred_norm, noise_norm)
    return changed_pred_norm - noise_norm * noise_factor


avg_score = visualize_and_evaluate_noise(gaussian_noise)

print(f"average score = {avg_score}")
