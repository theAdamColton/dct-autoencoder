import torch
from x_transformers import Encoder

dtype =torch.bfloat16
device='cuda'

encoder = Encoder(
        dim=768,
        depth=4,
        heads=32,
        attn_flash = True,
        use_rmsnorm = True,
        ff_glu = True,
        ff_no_bias = True,
        attn_one_kv_head = True,
        sandwich_norm = True,
        attn_qk_norm = True,
        attn_qk_norm_dim_scale = True,
        ).to(device).to(dtype)

x = 100 * torch.randn(16, 128, 768, dtype=dtype, device=device)
with torch.autocast(device, dtype=dtype):
    x = encoder(x)
assert not x.isnan().any().item()

