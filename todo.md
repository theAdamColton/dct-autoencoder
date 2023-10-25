# todo

* test if more wider images need more frequency information that is oriented left/right rather than frequency information that is oriented up/down

* fix color adherance
    * global pixel values are wildly varying, some images are painted very bright green or red

    * can manually reduce the chroma resolution, like in jpg
        * for an input patch of 3x16x16, I can split it up into 
        *  1x16x16 full res y channel,
        *  1x8x8 half res cb channel and
        *  1x8x8 half res cr channel
        *  and then flatten into 384 length patch vector

* try to see by what heuristic can patches can be dropped. If they are small mag.? If they have high frequency?

* add explicit pos embeddings for decoder, as to not force the vq codes to store the pos info
