import math
import os
import time
import project_paths as pp
from model import UNET, EncoderDecoder
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataloader import NoiseDataloader
from torch import optim

# --------------------------------------------------------------
# Hyperparameters
# --------------------------------------------------------------

pretrained_model_file_path = None
# pretrained_model_file_path = os.path.abspath(r"D:\Codes\Python\IIT Bombay\Semester 3\[CS 663] Fundamentals of Digital Image Processing\Course Project\trained models\Instance_004\Model_Epoch_000.pt")

MODEL = {
    'BATCH_SIZE': 20,
    'NUM_EPOCHS': 10,
    'NUM_WORKERS': 3
}
if torch.cuda.is_available():
    MODEL['DEVICE'] = 'cuda'
    torch.cuda.init()
    # backends.cudnn.benchmark = True
else:
    MODEL['DEVICE'] = 'cpu'

OPTIMIZER = {
    'LR': 0.001,
    'BETAS': (0.9, 0.99),
    'EPSILON': 1e-08,
    'LOSS_FUNCTION': torch.nn.L1Loss().to(MODEL['DEVICE'])
}

DATASET = {
    'NOISE_TYPE': NoiseDataloader.GAUSSIAN,
    'NOISY_PER_IMAGE': 300
}

# --------------------------------------------------------------


def train():
    # network = EncoderDecoder()
    network = UNET()
    network = nn.DataParallel(network)
    try:
        if pretrained_model_file_path != None and os.path.isfile(pretrained_model_file_path):
            network.load_state_dict(torch.load(pretrained_model_file_path))
            print('Network weights initialized from file at:', os.path.abspath(pretrained_model_file_path))
    except Exception:
        print('Unable to initialize network weights from file at:', os.path.abspath(pretrained_model_file_path))
    network.to(MODEL['DEVICE'])
    network.train()


    train_dataset = NoiseDataloader(dataset_type=NoiseDataloader.TRAIN,
                                    noisy_per_image=DATASET['NOISY_PER_IMAGE'],
                                    noise_type=DATASET['NOISE_TYPE'])

    train_batcher = DataLoader(dataset=train_dataset,
                               batch_size=MODEL['BATCH_SIZE'],
                               # shuffle=True)
                               shuffle = True,
                               num_workers = MODEL['NUM_WORKERS'])

    optimizer = optim.Adam(network.parameters(),
                           lr=OPTIMIZER['LR'],
                           betas=OPTIMIZER['BETAS'],
                           eps=OPTIMIZER['EPSILON'])

    instance = 0
    while os.path.isdir(os.path.join(pp.trained_models_folder_path, 'Instance_' + str(instance).zfill(3))):
        instance += 1
    os.mkdir(os.path.join(pp.trained_models_folder_path, 'Instance_' + str(instance).zfill(3)))

    num_batches = math.floor(len(train_dataset) / MODEL['BATCH_SIZE'])
    for epoch in range(MODEL['NUM_EPOCHS']):

        epoch_start_time = time.time()
        print('-' * 80)
        print('Epoch: {} of {}...'.format(epoch + 1, MODEL['NUM_EPOCHS']))

        epoch_loss = 0
        batch_counter = 1

        for batch in train_batcher:  # Get Batch
            print('\tProcessing Batch: {} of {}...'.format(batch_counter, num_batches))
            batch_counter += 1

            input_noisy_patch, output_noisy_patch = batch
            input_noisy_patch = input_noisy_patch.to(MODEL['DEVICE'])
            output_noisy_patch = output_noisy_patch.to(MODEL['DEVICE'])

            denoised_input_patch = network(input_noisy_patch)  # Pass Batch

            loss = OPTIMIZER['LOSS_FUNCTION'](denoised_input_patch, output_noisy_patch)  # Calculate Loss

            epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()  # Calculate Gradients
            optimizer.step()  # Update Weights
            print('\tBatch (Train) Loss:', loss)
            print()

        epoch_end_time = time.time()
        torch.save(network.state_dict(),
                   os.path.join(pp.trained_models_folder_path, 'Instance_' + str(instance).zfill(3), 'Model_Epoch_{}.pt'.format(str(epoch).zfill(3))))

        print('Epoch (Train) Loss:', epoch_loss)
        print('Epoch (Train) Time:', epoch_end_time - epoch_start_time)
        print('-' * 80)


if __name__ == '__main__':
    print('Commencing Training')
    train()
    print('Training Completed')
