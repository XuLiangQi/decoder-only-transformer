# Decoder Only Transformer
This project aims to create a generative decoder only transformer model from scratch by using pytorch.

## Dataset
This project uses open source text document [Tiny Shakespear](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt). Download the text file if you want to modify & train this model on your own.

## Installation
    conda env create -f environment.yml
This will create the env and all the necessary packages to run the scripts.

## To Start
The `main.py` is the main interface where the basic setting such as CUDA setup, program settings, and the training script is located.

To excute, run the line below:

    python main.py

## Note
There are two sets of hyperparameters under the `hyps` folder. The `hyps-small-model.yaml` is used primarily for development purpose, as the run time when using smaller scale parameters are faster.

To take advantage of CUDA capable GPUs, install [CUDA](https://developer.nvidia.com/cuda-downloads) from NVIDIA Developer website.
