# Sequential Denoising Autoencoder
A TensorFlow implementation of the SDAE model proposed in Hill et al. (2016).
(https://arxiv.org/abs/1602.03483)

## How to Train
Execute `python train.py` with appropriate options.
Note that `python train.py -h` gives you a list of available options.

## How to Generate Sentence Vectors
Execute `python encode.py` with appropriate options.

## Data Format
For both training and encoding, a data file should follow the below format.
```
sentence_1
sentence_2
sentence_3
...
sentence_N
```
Also note that tokens must be separated by spaces.
`python preprocess.py` does this chore.

## Version
Tested on TensorFlow v1.0.
