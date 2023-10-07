# DCT

DCT works very well using MLPs, that is:

Image -> image dct -> flattened image dct -> MLP -> vq -> MLP -> idct 

Using mse pixel loss works, also using mse freq space loss also works


DCT doesn't work very well with transformers. It would be nice to get it to
work with transformers because then it can be trained on images of different
dimensions.
