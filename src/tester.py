import torch
from torch import nn
from model import UNET, EncoderDecoder
import os
import project_paths as pp
from matplotlib import pyplot as plt
import numpy as np
from dataloader import NoiseDataloader
import random


def test(noise_type):
    global test_dataset
    if noise_type == NoiseDataloader.GAUSSIAN:
        test_dataset = NoiseDataloader(dataset_type=NoiseDataloader.TEST,
                                       noisy_per_image=1,
                                       noise_type=NoiseDataloader.GAUSSIAN)
    elif noise_type == NoiseDataloader.TEXT_OVERLAY:
        test_dataset = NoiseDataloader(dataset_type=NoiseDataloader.TEST,
                                       noisy_per_image=1,
                                       noise_type=NoiseDataloader.TEXT_OVERLAY)
    elif noise_type == NoiseDataloader.SALT_PEPPER:
        test_dataset = NoiseDataloader(dataset_type=NoiseDataloader.TEST,
                                       noisy_per_image=1,
                                       noise_type=NoiseDataloader.SALT_PEPPER)
    else:
        return

    # Initializing network
    network = EncoderDecoder()
    network = nn.DataParallel(network)
    instance = '010'
    pretrained_model_folder_path = os.path.join(pp.trained_models_folder_path, 'Instance_' + instance)
    for pretrained_model_file_name in os.listdir(pretrained_model_folder_path):
        try:
            if pretrained_model_file_name.endswith('.pt'):
                network.load_state_dict(
                    torch.load(os.path.join(pretrained_model_folder_path, pretrained_model_file_name)))
                print('Network weights initialized using file from:', pretrained_model_file_name)
            else:
                continue
        except:
            print('Unable to load network with weights from:', pretrained_model_file_name)
            continue

        idx = random.randint(0, len(test_dataset))
        noisy_image, clean_image = test_dataset[idx]
        predicted_image = network(torch.unsqueeze(torch.as_tensor(noisy_image), dim=0))[0]

        clean_image = NoiseDataloader.convert_model_output_to_image(clean_image)
        noisy_image = NoiseDataloader.convert_model_output_to_image(noisy_image)
        predicted_image = NoiseDataloader.convert_model_output_to_image(predicted_image)

        plt.figure(num='Network Performance using weights at {}'.format(pretrained_model_file_name), figsize=(20, 20))

        plt.subplot(2, 2, 1)
        plt.imshow(clean_image, cmap='gray')
        plt.colorbar()
        plt.title('Original Image')

        plt.subplot(2, 2, 2)
        plt.imshow(noisy_image, cmap='gray')
        plt.colorbar()
        plt.title('Noisy Image')

        plt.subplot(2, 2, 3)
        plt.imshow(predicted_image, cmap='gray')
        plt.colorbar()
        plt.title('Predicted Image')

        plt.subplot(2, 2, 4)
        plt.imshow(np.sqrt(np.sum((clean_image - predicted_image) ** 2, axis=2)), cmap='gray')
        plt.title('Euclidean Distance')
        plt.colorbar()
        plt.show()


if __name__ == '__main__':
    print('Commencing Testing')
    test(NoiseDataloader.GAUSSIAN)
    print('Testing Completed')
