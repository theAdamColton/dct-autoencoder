import time
import torch
from torch.utils.data import DataLoader
from dct_autoencoder import DCTAutoencoder
from dct_autoencoder.configuration_dct_autoencoder import DCTAutoencoderConfig
from dct_autoencoder.dataset import dict_collate
from dct_autoencoder.factory import get_model_and_processor
from dct_autoencoder.dataset import load_preprocessed_dataset

model_conf_path:str= "./conf/patch32-large.json"
model_config = DCTAutoencoderConfig.from_pretrained(model_conf_path)
model,proc=get_model_and_processor(model_config, device='cuda', dtype =torch.float16,sample_patches_beta = 0.007,)
model.patchnorm.frozen=True

model = torch.compile(model, dynamic=False, fullgraph=True)
model: DCTAutoencoder

dataset = load_preprocessed_dataset("/hdd/laion-improved-aesthetics-6p-preprocessed-p32-a0.007/000{000..430}.tar")
dl = DataLoader(dataset, batch_size=5, collate_fn=dict_collate, num_workers=1)
dl = proc.iter_batches(iter(dl), batch_size=5)

for batch in dl:
    batch = batch.to('cuda')
    batch.patches = batch.patches.to(torch.float16)
    batch = model.normalize_(batch)
    st = time.time()
    out = model(batch.shallow_copy())['dct_patches']
    et = time.time()
    print(et-st)
