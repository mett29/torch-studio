# GPT-2

This is a 1-to-1 replica of the original gpt-2 codebase, but written in PyTorch.

Original repo: https://github.com/openai/gpt-2/tree/master

### How to run

Download the encoder, hyperparameters, and vocabulary. Note that we're not downloading the weights or the checkpoints.

```bash
python gpt-2/download_model_config.py
```

```bash
python gpt-2/generate.py
```