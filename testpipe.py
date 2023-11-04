import torchvision
from dct_autoencoder import DCTAutoencoderFeatureExtractor
from dct_autoencoder.patchnorm import PatchNorm
from dct_autoencoder.util import imshow, tuple_collate
from dct_autoencoder.dct_patches import DCTPatches
import matplotlib.pyplot as plt
import random
import os

random.seed(42)

device = 'cuda'

image_files = os.listdir("./images/")
images = [
    torchvision.io.read_image("./images/" + image_file) / 255
    for image_file in image_files
        ]
images = [
        torchvision.transforms.Resize((512, 512))(image).to(device)
        for image in images]

max_patch_h=16
max_patch_w=16
patch_size = 16

proc = DCTAutoencoderFeatureExtractor(
        channels=3,
        patch_size=patch_size,
        sample_patches_beta=0.0005,
        max_patch_h=max_patch_h,
        max_patch_w=max_patch_w,
        max_seq_len=256,
        channel_importances=(8,1,1),
)

patchnorm = PatchNorm(
        max_patch_h=max_patch_h,
        max_patch_w=max_patch_w,
        patch_size=patch_size,
        channels=3,
)

preprocessed = [proc.preprocess(image) for image in images]

# collates
res = tuple_collate(preprocessed)

batch:DCTPatches = next(iter(proc.iter_batches(iter([res]), batch_size=None)))


patchnorm(batch)
batch.patches = patchnorm(batch)
print("std", batch.patches.std(dim=0).mean())
print("mean", batch.patches.mean())
batch.patches = patchnorm.inverse_norm(batch)

image = proc.postprocess(batch)[0]

imshow(image)
plt.savefig("junk.png")
print("saved junk.png")

