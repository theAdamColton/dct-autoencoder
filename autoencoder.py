import torch.nn.functional as F
import torch
import torch.nn as nn

from util import calculate_perplexity, get_upper_left_tri_p, get_upper_left_tri, inverse_fft, dct2, idct2


class Encoder(nn.Module):
    def __init__(self, image_channels: int = 3,):
        super().__init__()
        self.image_channels = image_channels
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


class VQAutoencoder(nn.Module):
    def __init__(self, vq_model, encoder, decoder,):
        super().__init__()

        self.vq_model = vq_model
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        # x: pixel values
        
        z = self.encoder(x)
        z, codes, commit_loss = self.vq_model(z)

        perplexity = calculate_perplexity(codes, self.vq_model.codebook_size)

        x_hat = self.decoder(z)

        rec_loss = F.mse_loss(x, x_hat)

        return dict(x_hat=x_hat, perplexity= perplexity, commit_loss= commit_loss, rec_loss= rec_loss, codes= codes, z= z, x=x)

class VQAutoencoderDCT(VQAutoencoder):
    def __init__(self, *args,  tri_n:int = 32, **kwargs):
        super().__init__(*args, **kwargs)
        self.tri_n = tri_n

    def forward(self, x):
        # x: pixel values
        x_dct = dct2(x, 'ortho')
        #mask = get_upper_left_tri(x_dct.shape, self.tri_n).to(x_dct.device).to(x_dct.dtype)
        #x_dct = x_dct * mask

        # shrinks
        x_dct_in = x_dct[..., :self.tri_n, :self.tri_n]

        z = self.encoder(x_dct_in)

        z, codes, commit_loss = self.vq_model(z)

        perplexity = calculate_perplexity(codes, self.vq_model.codebook_size)

        x_hat_dct = self.decoder(z)

        rec_loss = F.mse_loss(x_hat_dct, x_dct_in)

        # grows
        with torch.no_grad():
            x_hat_dct_grown = torch.zeros_like(x, requires_grad=False)
            x_hat_dct_grown[..., :self.tri_n, :self.tri_n] = x_hat_dct

            x_dct_compressed = torch.zeros_like(x_dct, requires_grad=False)
            x_dct_compressed[..., :self.tri_n, :self.tri_n] = x_dct_in

            x_hat = idct2(x_hat_dct_grown, 'ortho')
            x = idct2(x_dct_compressed, 'ortho')

        return dict(x_hat=x_hat, perplexity= perplexity, commit_loss= commit_loss, rec_loss= rec_loss, codes= codes, z= z, x=x)
