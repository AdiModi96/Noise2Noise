import numpy as np
import cv2
import os
import project_paths as pp
from model import UNET
import torch
from dataloader import NoiseDataloader
from torch import nn
from matplotlib import pyplot as plt


IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']

def denoise_using_bilateral(noisy_image):
    return cv2.bilateralFilter(noisy_image, 15, 2, 2)

def denoise_using_PCA(noisy_image):
    return noisy_image

def denosie_using_noise2noise(noisy_image):
    denoised_image = network(torch.unsqueeze(torch.as_tensor(NoiseDataloader.convert_image_to_model_input(noisy_image)), dim=0))[0]
    denoised_image = NoiseDataloader.convert_model_output_to_image(denoised_image)

    return denoised_image

pretrained_model_folder_path = os.path.join(pp.trained_models_folder_path, 'Instance_000', 'Model_Epoch_012.pt')
global network
network = UNET()
network = nn.DataParallel(network)
network.load_state_dict(torch.load(pretrained_model_folder_path))

# Custom Image
noise_type = 'Gaussian'
for image_file_name in os.listdir(os.path.join(pp.real_world_data, noise_type)):
    image_file_path = os.path.join(pp.real_world_data, noise_type, image_file_name)
    if os.path.isfile(image_file_path) and os.path.splitext(image_file_name)[1].lower() in IMAGE_EXTENSIONS:
        noisy_image = np.asarray(cv2.cvtColor(cv2.imread(image_file_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) / 255, dtype=np.float32)

        noise2noise_denoised_image = denosie_using_noise2noise(noisy_image)
        bilateral_denoised_image = denoise_using_bilateral(noisy_image)

        plt.figure(num='Comparison | Image: ' + image_file_name, figsize=(30, 10))

        plt.subplot(1, 3, 1)
        plt.title('Original Image')
        plt.imshow(noisy_image, cmap='gray')
        plt.colorbar()

        plt.subplot(1, 3, 2)
        plt.title('Denoised Using Noise2Noise')
        plt.imshow(noise2noise_denoised_image, cmap='gray')
        plt.colorbar()

        plt.subplot(1, 3, 3)
        plt.title('Denoised Using Bilateral Filtering')
        plt.imshow(bilateral_denoised_image, cmap='gray')
        plt.colorbar()

        # plt.subplot(2, 2, 4)
        # plt.title('Denoised Using PCA Based Denoising')
        # plt.imshow(noisy_image, cmap='gray')
        # plt.colorbar()

        plt.show()