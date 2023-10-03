import torch.nn.functional as F
import torch
import torch.nn as nn

from util import calculate_perplexity, dct2, idct2, flatten_zigzag, unflatten_zigzag, zigzag


class Encoder(nn.Module):
    def __init__(self, image_channels: int = 3, vq_channels: int = 64):
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
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.GELU(),

                nn.BatchNorm2d(32),
                nn.Conv2d(32, vq_channels, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                )

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, image_channels: int=3, vq_channels:int=64):
        super().__init__()

        self.layers = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Conv2d(vq_channels, 32, kernel_size=3, stride=1, padding=1),
                nn.GELU(),

                nn.BatchNorm2d(32),
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
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
    def __init__(self, image_channels: int, vq_model, vq_channels:int, vq_codes:int):
        super().__init__()

        assert vq_codes == 256

        self.vq_model = vq_model
        self.encoder = Encoder(image_channels, vq_channels)
        self.decoder = Decoder(image_channels, vq_channels)

    def forward(self, x):
        # x: pixel values
        
        z = self.encoder(x)
        z, codes, commit_loss = self.vq_model(z)

        perplexity = calculate_perplexity(codes, self.vq_model.codebook_size)

        x_hat = self.decoder(z)

        rec_loss = F.mse_loss(x, x_hat)

        return dict(x_hat=x_hat, perplexity= perplexity, commit_loss= commit_loss, rec_loss= rec_loss, codes= codes, z= z, x=x, pixel_loss=rec_loss)

class DCTAutoencoderTransformer(nn.Module):
    def __init__(self, image_channels: int, input_res: int, patch_size: int, n_layers:int= 2):
        """
        input_res: the square integer input resolution size.
        """
        super().__init__()
        self.n_patches = int(input_res**2 // patch_size)

        assert input_res**2 / patch_size == input_res**2 // patch_size

        self.image_channels = image_channels
        self.patch_size = patch_size
        self.pos_embed = nn.Embedding(self.n_patches, patch_size * image_channels)
        self.zigzag_i = nn.Parameter(zigzag(input_res, input_res), requires_grad=False)

        self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(patch_size * image_channels, 8, patch_size * 4, batch_first=True, norm_first=True),
                n_layers,
            )

        self.decoder =  nn.TransformerEncoder(
                nn.TransformerEncoderLayer(patch_size * image_channels, 8, patch_size * 4, batch_first=True, norm_first=True),
                n_layers,
            )

    def patchify(self, x):
        return flatten_zigzag(x, zigzag_indices=self.zigzag_i)

    def depatchify(self, x, h, w):
        return unflatten_zigzag(x, h, w, zigzag_indices=self.zigzag_i)

    def encode(self, x):
        b = x.shape[0]
        x = self.patchify(x)
        x = x.reshape(b, self.n_patches, self.patch_size * self.image_channels)
        x = x + self.pos_embed.weight
        x = self.encoder(x)
        return x

    def decode(self, x, h, w):
        x = self.decoder(x)
        b = x.shape[0]
        x = x.reshape(b, self.image_channels, self.n_patches * self.patch_size)
        return unflatten_zigzag(x, h, w)


class VQAutoencoderDCT(nn.Module):
    def __init__(self, image_channels: int, vq_model, dct_features:int = 32**2, vq_channels:int=32, vq_codes: int = 256):
        super().__init__()
        self.dct_features = dct_features
        self.vq_model = vq_model
        self.vq_model.accept_image_fmap = False
        self.vq_model.channel_last = True
        self.vq_channels = vq_channels
        self.vq_codes = vq_codes
        self.image_channels = image_channels

        #self.transformer = DCTAutoencoderTransformer(image_channels, dct_res, dct_res**2 // 16)

        # input is unnormalized features
        self.encoder = nn.Sequential(
                nn.LayerNorm(self.dct_features * image_channels),
                nn.Linear(self.dct_features * image_channels, vq_codes * vq_channels),
                nn.GELU(),
                )
        self.decoder = nn.Sequential(
                nn.LayerNorm(vq_codes * vq_channels),
                nn.Linear(vq_codes * vq_channels, self.dct_features * image_channels),
        )

    def forward(self, x):
        b, c, h, w = x.shape

        assert c == self.image_channels

        with torch.no_grad():
            # x: pixel values
            x_dct = dct2(x, 'ortho')

            # flattens in a zigzag pattern
            # shape b, c, self.dct_res^2
            if not hasattr(self, 'zigzag_i'):
                self.zigzag_i = zigzag(h, w).to(x.device)

            # takes self.dct_features count of lowest freq dct components
            x_dct_compressed_flat = flatten_zigzag(x_dct, self.zigzag_i)
            x_dct_compressed_flat[:,:,self.dct_features:] = 0.0
            # shape b,c,h,w
            x_dct_compressed = unflatten_zigzag(x_dct_compressed_flat, h, w, self.zigzag_i)
            # shape b,c,self.dct_features
            x_dct_compressed_flat = x_dct_compressed_flat[:,:,:self.dct_features]

        z = self.encoder(x_dct_compressed_flat.reshape(b, -1))
        z = z.reshape(b, self.vq_codes, self.vq_channels)

        z, codes, commit_loss = self.vq_model(z)

        with torch.no_grad():
            perplexity = calculate_perplexity(codes, self.vq_model.codebook_size)

        z = z.reshape(b, -1)

        # shape b, self.dct_features * image_channels
        x_hat_dct = self.decoder(z)
        x_hat_dct = x_hat_dct.reshape(b, self.image_channels, self.dct_features)

        # mse loss over dct features, which are both arranged in zigzag indexing
        rec_loss = F.mse_loss(x_hat_dct, x_dct_compressed_flat)

        # inverse dct
        with torch.no_grad():
            # grows to total dct features
            x_hat_dct_grown = torch.zeros_like(x_dct_compressed, requires_grad=False).reshape(b,c,h*w)
            # shape b,c,h*w
            x_hat_dct_grown[..., :self.dct_features] = x_hat_dct
            # shape b,c,h,w
            x_hat_dct_grown = unflatten_zigzag(x_hat_dct_grown, h,w,self.zigzag_i)

            x_hat = idct2(x_hat_dct_grown, 'ortho')
            x_compressed = idct2(x_dct_compressed, 'ortho')

            pixel_loss = F.mse_loss(x,x_hat)

        return dict(x_hat=x_hat, perplexity= perplexity, commit_loss= commit_loss, rec_loss= rec_loss, codes= codes, z= z, x=x_compressed, pixel_loss=pixel_loss)
