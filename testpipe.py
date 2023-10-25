import torchvision
from dct_autoencoder import DCTAutoencoderFeatureExtractor
from dct_autoencoder.util import imshow
import matplotlib.pyplot as plt
import random

random.seed(42)

image = torchvision.io.read_image("./images/girl.jpg") / 255
image = torchvision.transforms.Resize((512, 512))(image)


proc = DCTAutoencoderFeatureExtractor(
        channels=3,
        patch_size=16,
        sample_patches_beta=0.1,
        max_patch_h=32,
        max_patch_w=32,
        max_seq_len=256,
        channel_importances=(8,1,1),
)

res = proc.preprocess([image])

batch = next(iter(proc.iter_batches(iter([res]), batch_size=None)))

image = proc.postprocess(batch)[0]

imshow(image)
plt.savefig("junk.png")

