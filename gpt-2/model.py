from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass
class HParams:
    n_vocab: int = 0
    n_ctx: int = 1024
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12


def shape_list(x: Tensor):
    """
    Get the shape of a PyTorch tensor as a list of integers.
    """
    return list(x.size())

def softmax(x: Tensor, dim: int = -1):
    """
    Softmax function.
    """
    x = x - torch.max(x, dim=dim, keepdim=True)
    ex = torch.exp(x)
    return ex / torch.max(ex, dim=dim, keepdim=True)

def gelu(x: Tensor):
    """
    GeLU activation function.

    The 0.044715 part comes directly from the original paper
    as an approximation:
    https://arxiv.org/pdf/1606.08415
    """
    return 0.5*x*(1+torch.tanh(np.sqrt(2/torch.pi)*(x+0.044715*torch.pow(x,3))))

def norm(x: Tensor, *, dim: int = -1, epsilon: float = 1e-5):
    """
    Normalize to mean = 0, std = 1, then do a diagonal affine transform.
    """
    n_state = x.shape[-1]
    g = torch.ones(n_state)
    b = torch.zeros(n_state)
    u = torch.mean(x, dim=dim, keepdim=True)
    s = torch.mean(torch.square(x-u), dim=dim, keepdim=True)
    x = (x - u) * torch.rsqrt(s + epsilon)
    x = x * g + b
    return x

def split_states(x: Tensor, n: int):
    """
    Reshape the last dimension of x into [n, x.shape[-1]/n].
    The last dimension is divided into two smaller dimensions:
    n and x.shape[-1] // n.
    Example:
    Suppose x is a tensor of shape [32, 128, 512] and we want to
    split the last dimension into two dimensions with n = 8.
    This would reshape x into a tensor of shape [32, 128, 8, 64].
    """
    # Unpacking syntax: start holds all dims except the last one,
    # and m is the last dimension.
    *start, m = shape_list(x)
    return torch.reshape(x, start + [n, m//n])

def merge_states(x: Tensor):
    """
    Smash the last two dimensions of x into a single dimension.
    Essentially, split_states() reversed.
    """
    *start, a, b = shape_list(x)
    return torch.reshape(x, start + [a*b])

def conv1d(x: Tensor, nf: int, *, w_init_stdev: float = 0.02):
    """
    Custom implementation of a 1D convolution operation.

    Args:
        :param x: Tensor: The input tensor on which the 1D convolution will be applied.
        :param nf: int: The number of filters (also called output channels) for the convolution.
        :param w_init_stdev: float = 0.02: This is the initial value for the weights (w),
            which is set by default to 0.02. Instead of being a weight initializer like
            in neural networks, it simply fills the weight tensor with this constant value.
    """
    # nx represents the number of input channels/features
    *start, nx = shape_list(x)
    # Initialize weights
    # 1 as the first dimension ensures that the convolution operates
    # along the sequence dimension.
    w = torch.full([1, nx, nf], fill_value=w_init_stdev)
    # Initialize bias as a vector of zeros with size `nf`,
    # meaning that there's one bias value for each output channel.
    b = torch.zeros(nf)
    # Perform matrix multiplication
    c = torch.reshape(
        torch.matmul(
            torch.reshape(x, [-1, nx]), # Reshape input to 2D: [batch*length, nx]
            torch.reshape(w, [-1, nf])  # Reshape weights to 2D: [nx, nf]
        ) + b, # Add bias
        start + [nf] # Reshape the result back to original dimensions with `nf`
    )
    return c

def attention_mask(nd: int, ns: int, *, dtype):
    """
    This function generates a lower triangular matrix that is used as an attention mask.
    The mask ensures that position i in the sequence can only attend to positions ≤ i,
    which is essential when performing operations like causal (autoregressive) attention.

    Example:
    # i = [[0], [1], [2]]
    # j = [0, 1, 2, 3]

    # Result will be:
    tensor([[1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1]], dtype=torch.int32)

    Args:
        :param nd: Number of rows in the mask (typically, the length of the target sequence in decoding).
        :param ns: Number of columns in the mask (typically, the length of the source sequence in encoding).
        :param dtype: The data type to which the final mask is cast (e.g., torch.int32).
    """
    i = torch.arange(start=0, end=nd)[:, None] # Creates a column vector of shape [nd, 1]
    j = torch.arange(start=0, end=ns)          # Creates a row vector of shape [ns]
    m = i >= j - ns + nd                       # Generates the lower triangular mask
    return m.type(dtype)                       # Casts the mask to the specified dtype

def attn(x: Tensor, n_state: int, *, past: Tensor | None, hparams: HParams):
    """
    This function computes multi-head self-attention with
    the option to incorporate past memory.

    Args:
        :param x: Input tensor of shape [batch_size, sequence_length, features].
            Represents the input sequence embeddings on which attention will be computed.
        :param n_state: Dimensionality of the attention projection space. This represents the 
            number of features per head multiplied by the number of attention heads.
        :param past: A tensor of shape [batch_size, hparams.n_layer, 2, hparams.n_head,
            sequence, hparams.n_embd // hparams.n_head].
            Represents the past keys and values from previous steps of the attention mechanism.
            If provided, these past key-value pairs will be concatenated to the current ones
            to allow the model to "remember" previous states.

    Returns:
        Tuple[Tensor, Tensor]:
            - a: The output tensor after applying multi-head attention and projecting the result,
                of shape [batch_size, sequence_length, n_state].
            - present: A tensor containing the current keys and values for the attention mechanism, 
                of shape [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head].
                This can be used as input to 'past' in future calls to maintain continuity of memory across time steps.
    """
    assert x.dim() == 3  # Should be [batch, sequence, features]
    assert n_state % hparams.n_head == 0
    if past is not None:
        assert past.dim() == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    def split_heads(x: Tensor):
        # From [batch, sequence, features] to [batch, sequence, heads, features]
        x_split = split_states(x, hparams.n_head)
        # From [batch, sequence, heads, features] to [batch, heads, sequence, features]
        return x_split.permute([0, 2, 1, 3])

    def merge_heads(x: Tensor):
        # Reverse of split_heads
        return merge_states(x.permute([0, 2, 1, 3]))

    def mask_attn_weights(w: Tensor):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        # Reshape b to match the shape of the attention weights w
        # The first two dimensions ([1, 1]) are for broadcasting
        # across the batch and heads dimensions, ensuring the mask
        # is applied independently to each batch and attention head.
        b = torch.reshape(b, [1, 1, nd, ns])
        # w * b: This keeps the attention weights where b = 1 (i.e., for valid positions, meaning
        # the current and past tokens).
        # 1 - b: This creates a mask where b = 0, i.e., positions in the upper triangular part (future tokens).
        # large_negative_value * (1 - b) assigns a very large negative value (-1e10) to positions
        # in the upper triangular part (softmax will treat those positions as having near-zero probability).
        large_negative_value = torch.tensor(1e10, dtype=w.dtype)
        w = w * b - large_negative_value * (1 - b)
        return w

    def multihead_attn(q: Tensor, k: Tensor, v: Tensor):
        # q, k, v have shape [batch, heads, sequence, features]
        w = torch.matmul(q, k.t())
        # Attention weights are scaled by the inverse square root of the key's dimensionality
        d_k = v.size(-1)
        w = w / torch.sqrt(torch.tensor(d_k, dtype=w.dtype))
        w = mask_attn_weights(w)
        w = softmax(w)
        a = torch.matmul(w, v)
        return a

    # Increases the number of features by 3 times to then generate q, k, v
    c = conv1d(x, n_state*3)
    q, k, v = map(split_heads, torch.split(c, 3, dim=2))
    present = torch.stack([k, v], dim=1)
    if past is not None:
        # If past is provided, it means that there is previous
        # context from prior steps
        pk, pv = torch.unbind(past, dim=1)
        k = torch.cat([pk, k], dim=-2)
        v = torch.cat([pv, v], dim=-2)
    a = multihead_attn(q, k, v)
    a = merge_heads(a)
    # Project it back to the original number of features (n_state).
    # This step is similar to the final dense layer after attention
    # in a Transformer block.
    a = conv1d(a, n_state)
    return a, present

def mlp(x: Tensor, n_state: int):
    """
    2-layer fully connected neural network that applies
    a GELU activation after the first layer.
    """
    nx = x.size(-1)
    h = gelu(conv1d(x, n_state))
    h2 = conv1d(h, nx)
    return h2

def block(x: Tensor, *, past: Tensor | None, hparams: HParams):
    """
    Attention block:
    It first applies multi-head attention, adds the residual connection,
    and follows it up with an MLP block.
    """
    nx = x.size(-1)
    a, present = attn(norm(x), nx, past=past, hparams=hparams)
    x = x + a
    m = mlp(norm(x), nx*4)
    x = x + m
    return x, present

def past_shape(*, hparams: HParams, batch_size=None, sequence=None):
    return [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head]

def expand_tile(value: Tensor, size: int):
    """
    Add a new axis of the given size and repeat the tensor along that new axis.

    Args:
        value (Tensor): Input tensor to expand and tile.
        size (int): Size to repeat along the new axis.

    Returns:
        Tensor: The expanded and tiled tensor.
    """
    # Convert to tensor if not already a tensor
    value = torch.as_tensor(value)
    # Add a new dimension at the 0-th axis
    value = value.unsqueeze(0)
    # Repeat the tensor along the new dimension 'size' times.
    # The *([1] * value.dim()) ensures the tensor is repeated
    # only along the new dimension while keeping the other dimensions unchanged.
    return value.repeat(size, *([1] * value.dim()))

def positions_for(tokens: Tensor, past_length: int):
    """
    Computes positional encodings for input tokens based on their indices.
    It adds an offset to the positions using a past_length parameter.

    Example:
    - tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])  # Shape [2, 3] (batch size: 2, sequence length: 3)
    - past_length = 5
    - positions would be:
        tensor([[5, 6, 7],
                [5, 6, 7]])

    Args:
        :param tokens: Tensor of token indices with shape [batch_size, sequence_length].
        :param past_length: The number of previous tokens (i.e., offset for the positions).

    Returns:
        Tensor: The positions of the tokens adjusted by the past length, expanded for each batch.
    """
    batch_size = tokens.size(0)  # Get the batch size
    nsteps = tokens.size(1)      # Get the number of steps or sequence length
    # Use expand_tile() to repeat the positions for each batch
    return expand_tile(past_length + torch.arange(nsteps), batch_size)

def model(hparams: HParams, X: Tensor, past: Tensor | None = None):
    """
    Constructs a transformer model based on the given hyperparameters and input data.

    This function initializes the model parameters, performs embedding lookups for the input tokens,
    and processes the input through multiple transformer layers. It returns the computed logits for 
    language modeling as well as the present key-value pairs for attention mechanisms.

    Args:
        :param hparams: An object containing hyperparameters for the model, including:
            - n_ctx: The maximum context length for input sequences.
            - n_embd: The dimensionality of the embedding space.
            - n_vocab: The size of the vocabulary.
            - n_layer: The number of transformer layers to stack.
        :param X: A tensor of shape (batch_size, sequence_length) containing token IDs for the input.
        :param past: A tensor containing past key-value pairs for attention layers. 
            Shape should be
            [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head].
            If None, past attention context is not used.

    Returns:
        dict: A dictionary containing:
            - 'present': A tensor of shape
              [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head]
              containing the key-value pairs for each layer.
            - 'logits': A tensor of shape (batch_size, sequence_length, n_vocab) containing the predicted 
              token logits for the input sequences.
    """
    results = {}
    batch, sequence = shape_list(X)

    # Initialize positional and token embeddings with the specified shapes
    wpe = torch.randn(hparams.n_ctx, hparams.n_embd) * 0.01    # Positional Embeddings
    wte = torch.randn(hparams.n_vocab, hparams.n_embd) * 0.02  # Token Embeddings
    past_length = 0 if past is None else past.size(-2)
    # Gather token and positional embeddings
    h = wte[X] + wpe[positions_for(X, past_length)]

    # Initialize presents and pasts for the transformer layers
    presents = []
    pasts = past.unbind(dim=1) if past is not None else [None] * hparams.n_layer
    assert len(pasts) == hparams.n_layer
    # Loop through each transformer layer
    for layer, past in enumerate(pasts):
        h, present = block(h, past=past, hparams=hparams)
        presents.append(present)
    # Store the present values for all layers
    results['present'] = torch.stack(presents, dim=1)
    # Apply layer normalization to the final output
    h = norm(h)

    # Prepare the logits for language model loss
    h_flat = torch.reshape(h, [batch*sequence, hparams.n_embd])
    logits = torch.matmul(h_flat, wte.t())
    logits = torch.reshape(logits, [batch, sequence, hparams.n_vocab])
    # Store logits in results
    results['logits'] = logits
    return results


if __name__ == "__main__":
    # Debug
    print(attention_mask(nd=4, ns=4, dtype=torch.int32))
