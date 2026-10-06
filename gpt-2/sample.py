import model
import torch
from model import HParams
from torch import Tensor


def top_k_logits(logits: Tensor, k: int):
    """
    Truncate the logits by keeping only the top-k highest
    values, while setting the rest to a very low value.

    Args:
        :param logits: Logits of shape [batch, vocab_size].
        :param k: Number of top values to retain.

    Returns:
        torch.Tensor: Logits with only the top-k values retained, others set to -inf.
    """
    if k == 0:
        # no truncation
        return logits
    if not 0 <= k <= logits.size(-1):
        raise ValueError("top_k must be between 0 and the vocabulary size")

    def _top_k():
        values, _ = torch.topk(logits, k=k, dim=-1)
        # Get the smallest value among the top-k values in each row
        min_values = values[:, -1].unsqueeze(-1)
        return torch.where(
            logits < min_values,
            torch.full_like(logits, float("-inf")),
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
        torch.Tensor: Logits with only top-p elements retained, others set to -inf.
    """
    if not 0 <= p <= 1:
        raise ValueError("top_p must be between 0 and 1")
    if p == 1:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    # The softmax output is a normalized probability distribution over the vocabulary
    # torch.cumsum transforms e.g. [0.5, 0.3, 0.1, 0.05, 0.05] to [0.5, 0.8, 0.9, 0.95, 1.0]
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    # Keep the token that reaches the threshold, and always keep at least one.
    # Scatter the mask back to vocabulary order independently for each batch row.
    sorted_remove = cumulative_probs >= p
    sorted_remove = torch.cat(
        [torch.zeros_like(sorted_remove[:, :1]), sorted_remove[:, :-1]], dim=-1
    )
    remove = torch.zeros_like(sorted_remove).scatter(-1, sorted_indices, sorted_remove)
    return logits.masked_fill(remove, float("-inf"))


@torch.no_grad()
def sample_sequence(
    *,
    hparams: HParams,
    length: int,
    start_token: int = None,
    batch_size: int = None,
    context: Tensor = None,
    temperature: float = 1,
    top_k: int = 0,
    top_p: float = 1,
    parameters: dict | None = None,
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
        :param parameters: Optional dictionary to reuse model weights across samples.

    Returns:
        torch.Tensor: Generated sequence of tokens.
    """
    if length < 0:
        raise ValueError("length must be nonnegative")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 <= top_k <= hparams.n_vocab:
        raise ValueError("top_k must be between 0 and the vocabulary size")
    if not 0 <= top_p <= 1:
        raise ValueError("top_p must be between 0 and 1")
    if start_token is None:
        assert context is not None, "Specify exactly one of start_token and context!"
    else:
        assert context is None, "Specify exactly one of start_token and context!"
        if batch_size is None or batch_size <= 0:
            raise ValueError("batch_size must be positive when using start_token")
        context = torch.full([batch_size, 1], start_token, dtype=torch.long)
    if context.dim() != 2 or context.size(0) == 0 or context.size(1) == 0:
        raise ValueError(
            "context must have shape [batch, sequence] with nonempty dimensions"
        )
    if batch_size is not None and batch_size != context.size(0):
        raise ValueError("batch_size must match context")
    batch_size = context.size(0)
    # The final sampled token is returned without another model call.
    if context.size(1) + max(length - 1, 0) > hparams.n_ctx:
        raise ValueError("context and generation exceed the model window size")
    if parameters is None:
        parameters = {}

    def step(hparams, tokens, past=None):
        lm_output = model.model(
            hparams=hparams, X=tokens, past=past, parameters=parameters
        )

        logits = lm_output["logits"][:, :, : hparams.n_vocab]
        presents = lm_output["present"]
        sequence_length = tokens.size(1)
        past_shape = model.past_shape(
            hparams=hparams, batch_size=batch_size, sequence=sequence_length
        )
        presents = presents.reshape(past_shape)
        return {
            "logits": logits,
            "presents": presents,
        }

    def body(past, prev, output):
        next_outputs = step(hparams, prev, past=past)
        logits = next_outputs["logits"][:, -1, :] / temperature
        # Apply top_k and top_p sampling
        logits = top_k_logits(logits, k=top_k)
        logits = top_p_logits(logits, p=top_p)
        # Sample from the distribution
        samples = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        return [
            next_outputs["presents"]
            if past is None
            else torch.cat([past, next_outputs["presents"]], dim=-2),
            samples,
            torch.cat([output, samples], dim=1),
        ]

    past, prev, tokens = None, context, context
    for _ in range(length):
        past, prev, tokens = body(past, prev, tokens)

    return tokens
