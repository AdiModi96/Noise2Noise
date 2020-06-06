import torch
import torch.nn as nn


class UNET(nn.Module):
    input_channels = 3

    def __init__(self):
        super().__init__()

        self.encoder_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=UNET.input_channels, out_channels=48, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2)
        )

        self.encoder_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2)
        )

        self.encoder_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2)
        )

        self.encoder_block_4 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2)
        )

        self.encoder_block_5 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2)
        )

        self.encoder_convolution = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
        )

        self.decoder_upsample = nn.UpsamplingNearest2d(scale_factor=2)

        self.decoder_block_5 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.UpsamplingNearest2d(scale_factor=2)
        )

        self.decoder_block_4 = nn.Sequential(
            nn.Conv2d(in_channels=144, out_channels=96, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.UpsamplingNearest2d(scale_factor=2)
        )

        self.decoder_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=144, out_channels=96, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.UpsamplingNearest2d(scale_factor=2)
        )

        self.decoder_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=144, out_channels=96, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.UpsamplingNearest2d(scale_factor=2)
        )

        self.decoder_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=96 + UNET.input_channels, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(in_channels=32, out_channels=UNET.input_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        )

    def forward(self, tensor):
        # Encoder
        encoder_block_1_output = self.encoder_block_1(tensor)
        encoder_block_2_output = self.encoder_block_2(encoder_block_1_output)
        encoder_block_3_output = self.encoder_block_3(encoder_block_2_output)
        encoder_block_4_output = self.encoder_block_4(encoder_block_3_output)
        encoder_block_5_output = self.encoder_block_5(encoder_block_4_output)
        encoder_convolution_output = self.encoder_convolution(encoder_block_5_output)

        # Decoder
        decoder_upsample_output = self.decoder_upsample(encoder_convolution_output)
        decoder_block_5_output = self.decoder_block_5(torch.cat((decoder_upsample_output, encoder_block_4_output), dim=1))
        decoder_block_4_output = self.decoder_block_4(torch.cat((decoder_block_5_output, encoder_block_3_output), dim=1))
        decoder_block_3_output = self.decoder_block_3(torch.cat((decoder_block_4_output, encoder_block_2_output), dim=1))
        decoder_block_2_output = self.decoder_block_2(torch.cat((decoder_block_3_output, encoder_block_1_output), dim=1))
        decoder_block_1_output = self.decoder_block_1(torch.cat((decoder_block_2_output, tensor), dim=1))

        return decoder_block_1_output


class EncoderDecoder(nn.Module):
    input_channels = 3

    def __init__(self):
        super().__init__()

        self.encoder_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=EncoderDecoder.input_channels, out_channels=48, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2)
        )

        self.encoder_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2)
        )

        self.encoder_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2)
        )

        self.encoder_block_4 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2)
        )

        self.encoder_block_5 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2)
        )

        self.encoder_convolution = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
        )

        self.decoder_upsample = nn.UpsamplingNearest2d(scale_factor=2)

        self.decoder_block_5 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.UpsamplingNearest2d(scale_factor=2)
        )

        self.decoder_block_4 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.UpsamplingNearest2d(scale_factor=2)
        )

        self.decoder_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.UpsamplingNearest2d(scale_factor=2)
        )

        self.decoder_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.UpsamplingNearest2d(scale_factor=2)
        )

        self.decoder_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(in_channels=48, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(in_channels=32, out_channels=EncoderDecoder.input_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        )

    def forward(self, tensor):
        # Encoder
        tensor = self.encoder_block_1(tensor)
        tensor = self.encoder_block_2(tensor)
        tensor = self.encoder_block_3(tensor)
        tensor = self.encoder_block_4(tensor)
        tensor = self.encoder_block_5(tensor)
        tensor = self.encoder_convolution(tensor)

        # Decoder
        tensor = self.decoder_upsample(tensor)
        tensor = self.decoder_block_5(tensor)
        tensor = self.decoder_block_4(tensor)
        tensor = self.decoder_block_3(tensor)
        tensor = self.decoder_block_2(tensor)
        tensor = self.decoder_block_1(tensor)

        return tensor
