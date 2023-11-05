import os
import torchvision
import torch
from torch_dct import dct_2d, idct_2d
import matplotlib.pyplot as plt


from dct_autoencoder import DCTAutoencoderFeatureExtractor
from dct_autoencoder.util import (
    Tlms2rgb,
    Trgb2lms,
    rgb_to_ipt,
    ipt_to_rgb,
    rgb_to_lms,
    lms_to_rgb,
    imshow,
    MsRGB,
    MHPE,
    Mipt,
    channel_mult,
)


for image_file in os.listdir("./images/"):
    im_srgb = torchvision.io.read_image("./images/" + image_file) / 255
    im_srgb = torchvision.transforms.Resize(512)(im_srgb)
    c, h, w = im_srgb.shape

    noise = torch.randn(c, h, w) / 4 + 0.1

    im_xyz = channel_mult(
        MsRGB,
        im_srgb,
    )
    im_xyz_rec = channel_mult(MsRGB.inverse(), im_xyz)

    im_xyz = channel_mult(
        MsRGB,
        im_srgb,
    )
    im_xyz_rec = channel_mult(MsRGB.inverse(), im_xyz)
    im_xyz_rec_denoised = channel_mult(MsRGB.inverse(), im_xyz + noise)

    im_ipt = rgb_to_ipt(im_srgb)
    im_ipt_rec = ipt_to_rgb(im_ipt)
    im_ipt_rec_denoised = ipt_to_rgb(im_ipt + noise)

    f, axg = plt.subplots(3, 3)
    imshow(im_srgb, ax=axg[0][0])
    imshow(im_srgb, ax=axg[0][1])
    imshow(im_srgb + noise, ax=axg[0][2])
    imshow(im_xyz, ax=axg[1][0])
    imshow(im_xyz_rec, ax=axg[1][1])
    imshow(im_xyz_rec_denoised, ax=axg[1][2])
    imshow(im_ipt, ax=axg[2][0])
    imshow(im_ipt_rec, ax=axg[2][1])
    imshow(im_ipt_rec_denoised, ax=axg[2][2])

    plt.show()

    import bpdb

    bpdb.set_trace()
