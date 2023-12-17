from transformers.configuration_utils import PretrainedConfig
from transformers import CLIPVisionConfig


class DCTAutoencoderConfig(PretrainedConfig):
    def __init__(
        self,
        image_channels: int = 3,
        patch_size: int = 16,
        max_patch_h: int = 32,
        max_patch_w: int = 32,
        # VQ params
        vq_codebook_size: int = 4096,
        # number of heads, this is the number of codes per patch
        vq_num_codebooks: int = 8,
        # can be lfq or vq
        vq_type:str='lfq',

        encoder_config: CLIPVisionConfig = CLIPVisionConfig(),
        decoder_config: CLIPVisionConfig = CLIPVisionConfig(),
        **kwargs
    ):
        if not isinstance(encoder_config, CLIPVisionConfig):
            encoder_config = CLIPVisionConfig(**encoder_config)
        if not isinstance(decoder_config, CLIPVisionConfig):
            decoder_config = CLIPVisionConfig(**decoder_config)

        self.image_channels = image_channels
        self.patch_size = patch_size
        self.max_patch_h = max_patch_h
        self.max_patch_w = max_patch_w

        self.vq_type = vq_type

        self.vq_codebook_size = vq_codebook_size
        self.vq_num_codebooks = vq_num_codebooks

        self.encoder_config = encoder_config
        self.decoder_config = decoder_config

        super().__init__(**kwargs)
