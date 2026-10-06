# GPT-2

This is a minimal port of the original GPT-2 inference code to PyTorch.
It uses randomly initialized weights for learning purposes, so generated text
will not be coherent. Pretrained checkpoint loading is not implemented.

Original repo: https://github.com/openai/gpt-2/tree/master

### How to run

Download the encoder, hyperparameters, and vocabulary. Note that we're not downloading the weights or the checkpoints.

```bash
python gpt-2/download_model_config.py
```

```bash
python gpt-2/generate.py
```

When calling `model.model` directly, pass a shared `parameters={}` dictionary
to reuse embeddings and projection weights across calls. Cached `past` tensors
must be used with the parameters that produced them. `sample_sequence` manages
this dictionary automatically.

Run the regression checks with PyTorch, NumPy, and regex installed:

```bash
python -m unittest discover -s gpt-2 -p 'test_*.py'
```
