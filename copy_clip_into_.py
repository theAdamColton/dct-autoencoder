from transformers import CLIPVisionModel
from dct_autoencoder import DCTAutoencoder
from dct_autoencoder.configuration_dct_autoencoder import DCTAutoencoderConfig

def main(clip_url:str= "openai/clip-vit-large-patch14",
         model_conf_path:str= "./conf/patch32-large.json",
         out_path: str = "./out/clip_merged_model/"
         ):
    clip_vision = CLIPVisionModel.from_pretrained(clip_url)
    model_config = DCTAutoencoderConfig.from_pretrained(model_conf_path)
    model = DCTAutoencoder(model_config)

    assert clip_vision.config.hidden_size == model_config.encoder_config.hidden_size
    assert clip_vision.config.hidden_size == model_config.decoder_config.hidden_size

    clip_vision_encoder = clip_vision.vision_model.encoder
    n_clip_vision_layers = len(clip_vision_encoder.layers)
    n_encoder_vision_layers = len(model.encoder.layers)
    n_decoder_vision_layers = len(model.decoder.layers)

    clip_i = 0
    for i in range(n_encoder_vision_layers):
        model.encoder.layers[i] = clip_vision_encoder.layers[clip_i]
        clip_i += 1

    for i in range(n_decoder_vision_layers):
        model.decoder.layers[i] = clip_vision_encoder.layers[clip_i]
        clip_i += 1

    model.save_pretrained(out_path)

    print('done. saved to ', out_path)


if __name__ == "__main__":
    import jsonargparse
    jsonargparse.CLI(main)
