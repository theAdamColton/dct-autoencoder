# todo

* normalize dct input features based on some precomputed global dct statistics

* make the code sequence length invariant
    * change zigzag, to something else that allows scaling height and width of inputs;
    * more wider images need more frequency information that is oriented left/right rather than frequency information that is oriented up/down

* test flipped vs not flipped frequency ordering

* try making the encoder causal, and the decoder non-causal, what does this imply?

# 10/06/23

* change na\_vit so it doesn't do the query attention pooling I want a raw list of sequence length activations

* fix the h,w scaling
