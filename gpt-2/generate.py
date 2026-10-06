import json
import os

import encoder
import numpy as np
import sample
import torch
from model import HParams


def sample_model(
    seed=42,
    nsamples=0,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=0,
    top_p=1,
    config_dir=None,
):
    """
    Run the sample_model
    :param seed=None: Integer seed for random number generators, fix seed to
        reproduce results
    :param nsamples=0: Number of samples to return, if 0, continues to
        generate samples indefinately.
    :param batch_size=1: Number of batches (only affects speed/memory).
    :param length=None: Number of tokens in generated text, if None (default), is
        determined by model hyperparameters
    :param temperature=1: Float value controlling randomness in boltzmann
        distribution. Lower temperature results in less random completions. As the
        temperature approaches zero, the model will become deterministic and
        repetitive. Higher temperature results in more random completions.
    :param top_k=0: Integer value controlling diversity. 1 means only 1 word is
        considered for each step (token), resulting in deterministic completions,
        while 40 means 40 words are considered at each step. 0 (default) is a
        special setting meaning no restrictions. 40 generally is a good value.
    :param config_dir: path containing the encoder, vocabulary, and hyperparameters
    """
    if nsamples < 0 or batch_size <= 0:
        raise ValueError("nsamples must be nonnegative and batch_size must be positive")
    if config_dir is None:
        config_dir = os.path.join(os.path.dirname(__file__), "config")
    config_dir = os.path.expanduser(os.path.expandvars(config_dir))
    enc = encoder.get_encoder(config_dir)
    with open(os.path.join(config_dir, "hparams.json")) as f:
        hparams = HParams(**json.load(f))

    if length is None:
        length = hparams.n_ctx
    elif length > hparams.n_ctx:
        raise ValueError(f"Can't get samples longer than window size: {hparams.n_ctx}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    parameters = {}
    generated = 0
    while nsamples == 0 or generated < nsamples:
        current_batch_size = (
            batch_size if nsamples == 0 else min(batch_size, nsamples - generated)
        )
        output = sample.sample_sequence(
            hparams=hparams,
            length=length,
            start_token=enc.encoder["<|endoftext|>"],
            batch_size=current_batch_size,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            parameters=parameters,
        )[:, 1:]
        for i in range(current_batch_size):
            generated += 1
            text = enc.decode(output[i])
            print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
            print(text)


if __name__ == "__main__":
    """
    Generate
    """
    sample_model(top_k=40, nsamples=1, length=10)
