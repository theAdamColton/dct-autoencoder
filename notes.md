# DCT

DCT works very well using MLPs, that is:

Image -> image dct -> flattened image dct -> MLP -> vq -> MLP -> idct 

Using mse pixel loss works, also using mse freq space loss also works


DCT doesn't work very well with transformers. It would be nice to get it to
work with transformers because then it can be trained on images of different
dimensions.

# Processing pipeline

Image -> image dct -> patches

                        |||||||
                        vvvvvvv
                  batch of dct patches
                        |||||||
                        vvvvvvv
                 normalized dct patches   <- 
                        |||||||             |
                      transformer           |
                        \|||||/             |
                         vvvvv              |
                         codes              | freq space reconstruction loss
                        /|||||\             |
                        vvvvvvv             |
                          FF                |
                        |||||||             |
                        vvvvvvv             |
                 normalized dct patches   <-
                        |||||||
                        vvvvvvv
                 batch of dct patches
                           |
                           v
                        dct image -> image



