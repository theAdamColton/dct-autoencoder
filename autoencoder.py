import torch.nn.functional as F
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, image_channels: int = 3,):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Conv2d(image_channels, 16, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.GELU(),

                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.GELU(),

                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                )

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, image_channels: int=3,):
        super().__init__()

        self.layers = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
                nn.GELU(),

                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
                nn.GELU(),

                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Conv2d(16, image_channels, kernel_size=3, stride=1, padding=1),
                )

    def forward(self, x):
        return self.layers(x)


def calculate_perplexity(codes, codebook_size, null_index=-1):
    """
    Perplexity is 2^(H(p)) where H(p) is the entropy over the codebook likelyhood

    the null index is assumed to be -1, perplexity is only calculated over the
    non null codes
    """
    dtype, device = codes.dtype, codes.device
    codes = codes.flatten()
    codes = codes[codes!= null_index]
    src = torch.ones_like(codes)
    counts = torch.zeros(codebook_size).to(dtype).to(device)
    counts = counts.scatter_add_(0, codes, src)

    probs = counts / codes.numel()
    # Entropy H(x) when p(x)=0 is defined as 0
    logits = torch.log2(probs)
    logits[probs == 0.0] = 0.0
    entropy = -torch.sum(probs * logits)
    return 2**entropy


class VQAutoencoder(nn.Module):
    def __init__(self, vq_model, encoder:Encoder, decoder:Decoder):
        super().__init__()

        self.vq_model = vq_model
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        # x: pixel values
        x_fft = torch.fft.fft2(x)
        # concats in channel dimension
        x_fft = torch.concat([x_fft.real, x_fft.imag], dim=1)

        z = self.encoder(x_fft)
        z, codes, commit_loss = self.vq_model(z)

        perplexity = calculate_perplexity(codes, self.vq_model.codebook_size)

        x_hat_fft = self.decoder(z)

        rec_loss = F.mse_loss(x_fft, x_hat_fft)

        # splits into real,imag
        x_hat_fft = torch.complex(*torch.chunk(x_hat_fft, chunks=2, dim=1))

        x_hat = torch.fft.ifft2(x_hat_fft).real

        rec_loss = F.mse_loss(x, x_hat)

        return dict(x_hat=x_hat, perplexity= perplexity, commit_loss= commit_loss, rec_loss= rec_loss, codes= codes, z= z)

