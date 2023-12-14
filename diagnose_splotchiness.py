"""
what causes splotchiness? 
can it be remedied by simple noise filtering?
"""
import torch
import torchvision
from tqdm import tqdm

from dct_autoencoder.factory import get_model_and_processor
from dct_autoencoder.dataset import dict_collate, tuple_collate
from dct_autoencoder.util import image_clip

dtype = torch.float16
device = 'cuda'
image_path = "images/bold.jpg"
image = torchvision.io.read_image(image_path) / 255
image = image.to(dtype).to(device)
autoenc, proc = get_model_and_processor(resume_path="./out/2023-12-12_19-54-50/model/", device=device, dtype=dtype, sample_patches_beta=0.012)
input_data = proc.preprocess(image)
input_data = dict_collate([input_data])
                                                          
batch = next(iter(proc.iter_batches(iter([input_data]))))
batch = batch.to(device)
with torch.inference_mode():
    res = autoenc(batch, do_normalize=True)
