import torch
from torch.utils.data import Dataset
from torch import tensor
import cv2
import numpy as np
from typing import List
import os
import sys
import project_paths as pp

class NoiseDataloader(Dataset):

    IMAGE_EXTENSIONS = ['.jpg', '.png', '.jpeg']

    FONTS = [cv2.FONT_HERSHEY_COMPLEX,
             cv2.FONT_HERSHEY_COMPLEX_SMALL,
             cv2.FONT_HERSHEY_DUPLEX,
             cv2.FONT_HERSHEY_PLAIN,
             cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
             cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
             cv2.FONT_HERSHEY_SIMPLEX,
             cv2.FONT_HERSHEY_TRIPLEX]
    LINE_STYLES = [cv2.LINE_4,
                   cv2.LINE_8,
                   cv2.LINE_AA]

    ASCII = R''' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~'''
    ASCII_REDUCE = R'''0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'''

    TRAIN = 0
    TEST = 1
    VALIDATION = 2

    GAUSSIAN = 0
    POISSON = 1
    TEXT_OVERLAY = 2
    SALT_PEPPER = 3
    RANDOM_IMPULSE = 4

    PATCH_STRIDE = 256

    @staticmethod
    def crop(image):
        return image[0:-1, 0:-1]

    @staticmethod
    def convert_image_to_model_input(image):
        return image.transpose(2, 0, 1)

    @staticmethod
    def convert_model_output_to_image(torch_tensor):
        return torch.as_tensor(torch_tensor).to('cpu').detach().numpy().transpose(1, 2, 0)


    def __init__(self, dataset_type=TEST, noisy_per_image=500, noise_type=GAUSSIAN):

        self.dataset_type = dataset_type

        if dataset_type == NoiseDataloader.TRAIN:
            self.images_folder_path = os.path.join(pp.bsd_500_dataset_folder_path, 'train')
        if dataset_type == NoiseDataloader.TEST:
            self.images_folder_path = os.path.join(pp.bsd_500_dataset_folder_path, 'test')
        if dataset_type == NoiseDataloader.VALIDATION:
            self.images_folder_path = os.path.join(pp.bsd_500_dataset_folder_path, 'val')

        self.clean_patches: List[np.ndarray] = []

        if os.path.isdir(self.images_folder_path):
            for image_file_name in os.listdir(self.images_folder_path):
                image_file_path = os.path.join(self.images_folder_path, image_file_name)
                if os.path.isfile(image_file_path) and os.path.splitext(image_file_name)[1].lower() in NoiseDataloader.IMAGE_EXTENSIONS:
                    image = cv2.cvtColor(cv2.imread(image_file_path), cv2.COLOR_BGR2RGB)
                    for i in range(0, image.shape[0], NoiseDataloader.PATCH_STRIDE):
                        if i + NoiseDataloader.PATCH_STRIDE >= image.shape[0]:
                            i = image.shape[0] - NoiseDataloader.PATCH_STRIDE
                        for j in range(0, image.shape[1], NoiseDataloader.PATCH_STRIDE):
                            if j + NoiseDataloader.PATCH_STRIDE >= image.shape[1]:
                                j = image.shape[1] - NoiseDataloader.PATCH_STRIDE

                            patch = image[i:i + NoiseDataloader.PATCH_STRIDE, j:j + NoiseDataloader.PATCH_STRIDE]
                            self.clean_patches.append(patch)
        else:
            print("Dataset Path Doesn't Exist!")
            sys.exit(0)
        self.number_of_clean_patches = len(self.clean_patches)
        self.noisy_per_patch = noisy_per_image
        self.noise_type = noise_type


    def __len__(self):
        return self.noisy_per_patch * self.number_of_clean_patches


    def __getitem__(self, idx):
        clean_patch = self.clean_patches[int(idx) % self.number_of_clean_patches]

        if self.dataset_type != NoiseDataloader.TRAIN:
            noisy_image = NoiseDataloader.add_noise(clean_patch, self.noise_type)
            return (NoiseDataloader.convert_image_to_model_input(np.asarray(noisy_image / 255, dtype=np.float32)),
                    (NoiseDataloader.convert_image_to_model_input(np.asarray(clean_patch / 255, dtype=np.float32))))
        else:
            noisy_image_1 = NoiseDataloader.add_noise(clean_patch, self.noise_type)
            noisy_image_2 = NoiseDataloader.add_noise(clean_patch, self.noise_type)
            return (NoiseDataloader.convert_image_to_model_input(np.asarray(noisy_image_1 / 255, dtype=np.float32)),
                    (NoiseDataloader.convert_image_to_model_input(np.asarray(noisy_image_2 / 255, dtype=np.float32))))

    @staticmethod
    def add_noise(clean_patch, noise_type):

        global noisy_image
        if noise_type == NoiseDataloader.GAUSSIAN:
            # Adding Zero Mean Gaussian Noise
            noisy_image = clean_patch + np.random.normal(loc=0.0, scale=25, size=clean_patch.shape)

        elif noise_type == NoiseDataloader.POISSON:
            # Adding Poisson Noise
            noisy_image = np.random.poisson(clean_patch)

        elif noise_type == NoiseDataloader.TEXT_OVERLAY:
            noisy_image = NoiseDataloader.text_noise(clean_patch)

        elif noise_type == NoiseDataloader.SALT_PEPPER:
            noisy_image = NoiseDataloader.salt_pepper_noise(clean_patch)

        elif noise_type == NoiseDataloader.RANDOM_IMPULSE:
            noisy_image = NoiseDataloader.random_impulse_noise(clean_patch)

        return noisy_image

    @staticmethod
    def random_str(length):
        ascii_chars = [i for i in NoiseDataloader.ASCII_REDUCE]
        ascii_chars_np = np.array(ascii_chars)
        string = np.random.choice(ascii_chars_np, length)
        return "".join(list(string))

    @staticmethod
    def text_noise(img):
        num_str = np.random.randint(3, 10)
        noise = img.copy()
        x, y = img.shape[0], img.shape[1]

        for i in range(num_str):
            string = NoiseDataloader.random_str(np.random.randint(3, 20))
            font = np.random.choice(NoiseDataloader.FONTS)
            line_style = np.random.choice(NoiseDataloader.LINE_STYLES)
            font_size = np.random.uniform(2, 4)
            col = tuple(np.random.randint(0, 255, 3).astype(np.float64))
            # thickness = np.random.randint(3, 10)
            pos = (np.random.randint(0-x/100, x-x/50), np.random.randint(0-y/100, y-y/25))
            noise = cv2.putText(noise, string, pos, font, font_size, col, 3, line_style)

        return noise

    @staticmethod
    def index_1d_to_2d(i, y):
        return i // y, i % y

    @staticmethod
    def salt_pepper_noise(img, black_ratio=0.05, white_ratio=0.05):

        noise = img.copy()

        total = black_ratio + white_ratio
        x, y = img.shape[0], img.shape[1]

        indexes = np.random.choice(np.arange(x*y), size=int(total*x*y),
                                   replace=False)
        b_indexes = np.random.choice(indexes, size=int(black_ratio*x*y),
                                     replace=False)
        w_indexes = np.random.choice(indexes, size=int(white_ratio*x*y),
                                     replace=False)

        vector_index = np.vectorize(lambda i: NoiseDataloader.index_1d_to_2d(i, y))

        br, bc = vector_index(b_indexes)
        noise[br, bc] = np.array([0, 0, 0])
        wr, wc = vector_index(w_indexes)
        noise[wr, wc] = np.array([255, 255, 255])

        return noise

    @staticmethod
    def random_colour():
        return np.random.randint(0, 255, 3)

    @staticmethod
    def random_impulse_noise(img, ratio=0.4):
        noise = img.copy()

        x, y = img.shape[0], img.shape[1]

        indexes = np.random.choice(np.arange(x*y), size=int(ratio*x*y),
                                   replace=False)

        vector_index = np.vectorize(lambda i: NoiseDataloader.index_1d_to_2d(i, y))

        r, c = vector_index(indexes)
        noise[r, c] = np.random.randint(0, 256, noise[r, c].shape)
        # print(noise[wr, wc])

        return noise