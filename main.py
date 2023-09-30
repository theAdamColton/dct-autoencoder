from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch
import vector_quantize_pytorch
import datasets
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms

from autoencoder import Encoder, Decoder


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


def train(vq_model:nn.Module, encoder:nn.Module, decoder:nn.Module, train_ds, codebook_size:int, batch_size:int = 32, alpha: float = 1e-2, learning_rate=1e-8, epochs: int = 1, image_channels: int = 3, device='cuda'):
    dataloader = DataLoader(train_ds, batch_size=batch_size, )
    optimizer = Adam([*vq_model.parameters(), *encoder.parameters(), *decoder.parameters()], lr=learning_rate)

    for epoch in range(epochs):
        for batch in tqdm(dataloader):
            optimizer.zero_grad()
            x = batch['pixel_values']

            x_fft = torch.fft.fft2(batch['pixel_values'])
            # concats in channel dimension
            x_fft = torch.concat([x_fft.real, x_fft.imag], dim=1)

            with torch.autocast(device):
                z = encoder(x_fft)
                z, codes, commit_loss = vq_model(z)

                perplexity = calculate_perplexity(codes, codebook_size)

                x_hat_fft = decoder(z)

                # splits into real,imag
                x_hat_fft = torch.complex(*torch.chunk(x_hat_fft, chunks=2, dim=1))

                x_hat = torch.fft.ifft2(x_hat_fft).real

                rec_loss = F.mse_loss(x, x_hat)

                loss = rec_loss + alpha * commit_loss

            loss.backward()
            optimizer.step()

            print(f"epoch: {epoch} loss: {loss.item():.3f} rec_loss: {rec_loss.item():.2f} commit_loss: {commit_loss.item():.2f} perpelxity: {perplexity.item():.2f}")

    return vq_model, encoder, decoder

def load_and_transform_dataset(dataset_name_or_url: str, split:str, image_channels: int = 3, height:int=128, width:int=128, device='cuda'):
    ds = datasets.load_dataset(dataset_name_or_url, split=split)
    def f(examples):
        # norm parameters taken from clip
        _transforms = transforms.Compose([transforms.CenterCrop((height, width)), transforms.ToImageTensor(), transforms.ConvertImageDtype(torch.float32), transforms.Normalize((0.48145466,0.4578275,0.40821073), (0.26862954,0.26130258,0.27577711))])
        examples["pixel_values"] = [_transforms(image.convert("RGB")) for image in examples["image"]]
        return examples

    ds = ds.map(f, remove_columns=['image'], batched=True).with_format('torch', device=device)
    pixel_features = datasets.Array3D(shape=(image_channels, height, width), dtype='float32')
    ds.features['pixel_values'] = pixel_features
    ds = ds
    return ds


def main(image_dataset_path_or_url="imagenet-1k", device='cuda'):
    ds_train = load_and_transform_dataset(image_dataset_path_or_url, split='train', device=device)
    ds_test = load_and_transform_dataset(image_dataset_path_or_url, split='test', device=device)
    dtype = torch.float16

    codebook_sizes = [2**i for i in range(2, 16, 2)]

    autoencoder_depth = 3
    autoencoder_channel_mult = 4
    image_channels = 3
    input_channels = image_channels * 2 # 3 real number channels, 3 imaginary number channels

    for codebook_size in codebook_sizes:
        encoder = Encoder(autoencoder_depth, input_channels, autoencoder_channel_mult).to(device)
        decoder = Decoder(autoencoder_depth, input_channels, autoencoder_channel_mult).to(device)
        vq_model = vector_quantize_pytorch.VectorQuantize(autoencoder_channel_mult ** (autoencoder_depth+1), codebook_size, channel_last=False, accept_image_fmap=True, kmeans_init=True).to(device)

        print("training vq model")
        vq_model, encoder, decoder = train(vq_model, encoder, decoder, ds_train, codebook_size, device=device)

if __name__ == "__main__":
    import jsonargparse
    jsonargparse.CLI(main)
