from tqdm import tqdm
import torch
import vector_quantize_pytorch
import datasets
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
import os
import math

from autoencoder import VQAutoencoder, VQAutoencoderDCT
from util import imshow

import matplotlib.pyplot as plt
import wandb
from PIL import Image

IMAGEDIR="out/"

os.makedirs(IMAGEDIR, exist_ok=True)



def log_images(vq_autoencoder, x,filename:str, n:int=10):
    with torch.no_grad():
        out = vq_autoencoder(x[:n])
    fig, axarr = plt.subplots(n,2)


    for i in range(n):
        im = out['x'][i]
        im_hat = out['x_hat'][i]
        imshow(im.cpu(), axarr[i,0])
        imshow(im_hat.cpu(), axarr[i,1])

    plt.savefig(IMAGEDIR + "/" + filename)

    im=  Image.frombytes('RGB',fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
    plt.close()
    return im


def validate(vq_autoencoder, val_ds, batch_size: int=32, device='cuda', dtype=torch.float16):
    vq_autoencoder.eval()
    dataloader = DataLoader(val_ds, batch_size=batch_size, num_workers=8)
    for i, batch in enumerate(tqdm(dataloader)):
        x = batch['pixel_values']
        x = x.to(device)
        with torch.no_grad():
            with torch.autocast(device):
                out = vq_autoencoder(x)
        wandb.log({'val':dict(step=i, rec_loss=out['rec_loss'].item(), commit_loss=out['commit_loss'].item(), perplexity=out['perplexity'].item(), pixel_loss=out['pixel_loss'].item())})

        if i==0:
            image = log_images(vq_autoencoder, x, filename="validation.jpg")
            wandb.log({"val":dict(image=wandb.Image(image))})
        

def train(vq_autoencoder, train_ds, batch_size:int = 32, alpha: float = 5e-1, learning_rate=6e-4, epochs: int = 1, device='cuda', dtype=torch.float16, log_every = 20, log_info:dict = {}, max_steps=1000):

    optimizer = torch.optim.Adam(vq_autoencoder.parameters(), lr=learning_rate,)

    n_steps = 0

    for epoch in range(epochs):
        train_ds = train_ds.shuffle()
        dataloader = DataLoader(train_ds, batch_size=batch_size, drop_last=True, num_workers=8)
        for i, batch in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()
            x = batch['pixel_values']
            x = x.to(device)

            with torch.autocast(device, dtype=dtype):
                out = vq_autoencoder(x)

            commit_loss = out['commit_loss'] * alpha

            loss = out['rec_loss'] + commit_loss

            loss.backward()
            optimizer.step()


            print(f"epoch: {epoch} loss: {loss.item():.3f} rec_loss: {out['rec_loss'].item():.2f} commit_loss: {commit_loss.item():.2f} perpelxity: {out['perplexity'].item():.2f}")

            wandb.log({'train':dict(epoch=epoch, step=i, loss=loss.item(), rec_loss=out['rec_loss'].item(), commit_loss=commit_loss.item(), perplexity=out['perplexity'].item(), pixel_loss=out['pixel_loss'].item())})

            if i % log_every == 0:
                image = log_images(vq_autoencoder, x, filename=str(log_info))
                wandb.log({"train":dict(epoch=epoch, step=i, image=wandb.Image(image))})

            if n_steps > max_steps:
                return vq_autoencoder

            n_steps += 1

    wandb.finish()
    return vq_autoencoder


def unnormalize():
    return transforms.Compose([ transforms.Normalize(( 0., 0., 0. ),(1/0.26862954,1/0.26130258,1/0.27577711)),
                                transforms.Normalize((-0.48145466,-0.4578275,-0.40821073) ,( 1., 1., 1. )),
                               ])

def load_and_transform_dataset(dataset_name_or_url: str, split:str, image_channels: int = 3, height:int=128, width:int=128, rand:bool=True):
    ds = datasets.load_dataset(dataset_name_or_url, split=split)
    def f(examples):
        # norm parameters taken from clip
        _transforms = transforms.Compose([transforms.RandomResizedCrop((height, width)) if rand else transforms.CenterCrop((height,width)), transforms.ToImageTensor(), transforms.ConvertImageDtype(torch.float32), transforms.Normalize((0.48145466,0.4578275,0.40821073), (0.26862954,0.26130258,0.27577711))])
        return {'pixel_values':[_transforms(image.convert("RGB")) for image in examples["image"]]}

    ds.set_transform(f)
    pixel_features = datasets.Array3D(shape=(image_channels, height, width), dtype='float32')
    ds.features['pixel_values'] = pixel_features
    ds = ds.shuffle()
    return ds


def main(image_dataset_path_or_url="imagenet-1k", device='cuda', batch_size:int=32):
    ds_train = load_and_transform_dataset(image_dataset_path_or_url, split='train', rand=True)
    ds_test = load_and_transform_dataset(image_dataset_path_or_url, split='test', rand=False)
    dtype = torch.float16

    image_channels = 3
    input_channels = image_channels
    vq_channels = 64
    vq_codes = 256

    def get_vq_autoencoder(use_dct, codebook_size, heads):
        vq_model = vector_quantize_pytorch.VectorQuantize(vq_channels, codebook_size=codebook_size, codebook_dim=16, threshold_ema_dead_code=1, heads=heads, channel_last=False, accept_image_fmap=True, kmeans_init=True, separate_codebook_per_head=False, sample_codebook_temp=1.0).to(device)

        if use_dct:
            return VQAutoencoderDCT(input_channels, vq_model, vq_channels=vq_channels, vq_codes=vq_codes).to(device)
        return VQAutoencoder(input_channels, vq_model, vq_channels=vq_channels, vq_codes=vq_codes).to(device)

    codebook_sizes = [2**i for i in range(5, 11, 2)]

    head_numbers = [1, 4, 8]

    for codebook_size in codebook_sizes:
        for heads in head_numbers:
            for use_dct in [False, True]:
                vq_autoencoder = get_vq_autoencoder(use_dct, codebook_size, heads)

                run_d = dict(use_dct = use_dct, codebook_size=codebook_size, heads=heads, bits=vq_codes * heads * math.log2(codebook_size))

                print('starting run: ', run_d)

                #with torch.no_grad():
                #    with torch.autocast(device):
                #        vq_autoencoder(torch.randn(batch_size, 3, 128, 128).to(device).to(dtype))
                #continue

                run = wandb.init(project="vq-experiments", config=run_d)

                vq_autoencoder = train(vq_autoencoder, ds_train, device=device, dtype=dtype, batch_size=batch_size)
                validate(vq_autoencoder, ds_test, device=device, dtype=dtype, batch_size=batch_size)

                vq_autoencoder = None

                wandb.finish()


if __name__ == "__main__":
    import jsonargparse
    jsonargparse.CLI(main)
