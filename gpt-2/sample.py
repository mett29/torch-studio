import torch
from torch import Tensor

import model
from model import HParams

def top_k_logits(logits: Tensor, k: int):
    """
    Truncate the logits by keeping only the top-k highest
    values, while setting the rest to a very low value.

    Args:
        :param logits: Logits of shape [batch, vocab_size].
        :param k: Number of top values to retain.

    Returns:
        torch.Tensor: Logits with only the top-k values retained, others set to -1e10.
    """
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = torch.topk(logits, k=k, dim=-1)
        # Get the smallest value among the top-k values in each row
        min_values = values[:, -1].unsqueeze(-1)
        return torch.where(
            logits < min_values,
            torch.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )

    return _top_k()

def top_p_logits(logits: Tensor, p: float):
    """
    Nucleus sampling.
    In nucleus sampling, instead of selecting the top-k logits,
    we select the smallest subset of logits whose cumulative probability
    adds up to a predefined threshold p (typically around 0.9 or 0.95).
    This ensures that a dynamic number of tokens (instead of a fixed k)
    are selected for sampling based on their cumulative probabilities.

    Args:
        :param logits: Logits of shape [batch, vocab_size].
        :param p: Cumulative probability threshold for nucleus sampling.

    Returns:
        torch.Tensor: Logits with only top-p elements retained, others set to -1e10.
    """
    # logits has shape [batch, vocab_size]
    batch_size, _ = logits.shape
    sorted_logits, _ = torch.sort(logits, descending=True, dim=-1)
    # The softmax output is a normalized probability distribution over the vocabulary
    # torch.cumsum transforms e.g. [0.5, 0.3, 0.1, 0.05, 0.05] to [0.5, 0.8, 0.9, 0.95, 1.0]
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    # Create indices for each batch
    indices = torch.stack([
        torch.arange(batch_size),
        # number of indices to include
        torch.max(
            torch.sum((cumulative_probs <= p).type(torch.int32), dim=-1) - 1,
            torch.zeros(batch_size, dtype=torch.long)
        )
    ], dim=-1)
    # Gather the minimum logit values to retain
    min_values = sorted_logits[indices[:, 0], indices[:, 1]]
    return torch.where(
        logits < min_values,
        torch.ones_like(logits) * -1e10,
        logits,
    )

def sample_sequence(
    *, hparams: HParams, length: int, start_token: int = None, batch_size: int = None,
    context: Tensor = None, temperature: float = 1, top_k: int = 0, top_p: float = 1
):
    """
    Generates a sequence of tokens using the model and the given hyperparameters.
    
    Args:
        :param hparams: Model hyperparameters.
        :param length: The length of the sequence to generate.
        :param start_token: The token to start the sequence. Specify this or 'context'.
        :param batch_size: The batch size for generation.
        :param context: Initial context of tokens.
        :param temperature: Temperature for sampling.
        :param top_k: If > 0, only the top k tokens will be considered.
        :param top_p: If < 1, use top-p (nucleus) sampling.
    
    Returns:
        torch.Tensor: Generated sequence of tokens.
    """
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = torch.full([batch_size, 1], start_token, dtype=torch.long)

    def step(hparams, tokens, past=None):
        lm_output = model.model(hparams=hparams, X=tokens, past=past)

        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        presents = lm_output['present']
        presents.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))
        return {
            'logits': logits,
            'presents': presents,
        }

    def body(past, prev, output):
        next_outputs = step(hparams, prev, past=past)
        logits = next_outputs['logits'][:, -1, :] / temperature
        # Apply top_k and top_p sampling
        logits = top_k_logits(logits, k=top_k)
        logits = top_p_logits(logits, p=top_p)
        # Sample from the distribution
        samples = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        return [
            next_outputs['presents'] if past is None else torch.cat([past, next_outputs['presents']], dim=-2),
            samples,
            torch.cat([output, samples], dim=1)
        ]

    past, prev, tokens = body(None, context, context)

    for _ in range(length - 1):
        past, prev, tokens = body(past, prev, tokens)

    return tokens
