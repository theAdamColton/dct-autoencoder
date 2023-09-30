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

                nn.BatchNorm2d(16),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.GELU(),

                nn.BatchNorm2d(32),
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

                nn.BatchNorm2d(32),
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
                nn.GELU(),

                nn.BatchNorm2d(16),
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

def inverse_fft(fft_amp, fft_pha):
    imag = fft_amp * torch.sin(fft_pha)
    real = fft_amp * torch.cos(fft_pha)
    fft_y = torch.complex(real, imag)
    y = torch.fft.ifft2(fft_y)
    return y

def gaussian_fn(M, std):
    n = torch.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-n ** 2 / sig2)
    return w

def gkern(size=256, std=None):
    """Returns a 2D Gaussian kernel array."""
    if std is None:
        std = size / 2
    gkern1d = gaussian_fn(size, std=std) 
    gkern2d = torch.outer(gkern1d, gkern1d)
    return gkern2d


class VQAutoencoder(nn.Module):
    def __init__(self, vq_model, encoder, decoder, use_fft: bool):
        super().__init__()

        self.vq_model = vq_model
        self.encoder = encoder
        self.decoder = decoder
        self.use_fft = use_fft

    def forward_fft(self, x):

        # x: pixel values
        
        x_fft = torch.fft.fft2(x)
        # concats in channel dimension
        x_fft_amp = torch.sqrt(x_fft.real ** 2 + x_fft.imag ** 2)
        x_fft_phase = torch.atan2(x_fft.imag, x_fft.real)
        x_feat = torch.concat([x_fft_amp, x_fft_phase,], dim=1)

        z = self.encoder(x_feat)
        z, codes, commit_loss = self.vq_model(z)

        perplexity = calculate_perplexity(codes, self.vq_model.codebook_size)

        x_hat_fft = self.decoder(z)
        x_hat_fft_amp, x_hat_fft_phase = x_hat_fft.chunk(2, dim=1)

        rec_loss_amp = F.mse_loss(x_fft_amp.abs(), x_hat_fft_amp.abs())
        rec_loss_phase = F.mse_loss(x_fft_phase, x_hat_fft_phase)

        rec_loss = (rec_loss_amp + rec_loss_phase) / 2.0

        x_hat = inverse_fft(x_hat_fft_amp, x_hat_fft_phase).real
        x = inverse_fft(x_fft_amp, x_fft_phase).real

        #rec_loss = F.mse_loss(x, x_hat)

        return dict(x_hat=x_hat, perplexity= perplexity, commit_loss= commit_loss, rec_loss= rec_loss, codes= codes, z= z, x=x)


    def forward(self,x):
        if self.use_fft:
            return self.forward_fft(x)
        else:
            return self.forward_pixels(x)

    def forward_pixels(self, x):
        # x: pixel values
        
        z = self.encoder(x)
        z, codes, commit_loss = self.vq_model(z)

        perplexity = calculate_perplexity(codes, self.vq_model.codebook_size)

        x_hat = self.decoder(z)

        rec_loss = F.mse_loss(x, x_hat)

        return dict(x_hat=x_hat, perplexity= perplexity, commit_loss= commit_loss, rec_loss= rec_loss, codes= codes, z= z, x=x)

