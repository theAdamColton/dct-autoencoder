from transformers.configuration_utils import PretrainedConfig

class DCTAutoencoderConfig(PretrainedConfig):
    def __init__(
            self,
            image_channels: int = 3,
            n_layers: int = 4,
            n_attention_heads:int=16,
            feature_dim:int = 64,
            patch_size: int = 4,
            max_n_patches: int = 512,
            channel_diversity_loss_coeff: float = 1e0,

            # VQ params
            vq_codebook_size:int = 4096,
            # number of heads, this is the number of codes per patch
            vq_num_codebooks:int = 8,
            **kwargs):
        self.image_channels=image_channels
        self.n_layers = n_layers
        self.n_attention_heads = n_attention_heads
        self.feature_dim = feature_dim
        self.patch_size = patch_size
        self.max_n_patches = max_n_patches
        self.channel_diversity_loss_coeff = channel_diversity_loss_coeff

        self.vq_codebook_size = vq_codebook_size
        self.vq_num_codebooks = vq_num_codebooks

        super().__init__(**kwargs)
