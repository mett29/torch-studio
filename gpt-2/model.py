import numpy as np
import torch
from torch import Tensor


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

def attn(x: Tensor, n_state: int, *, past: Tensor, hparams):
    """
    TODO
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






if __name__ == "__main__":
    # Debug
    print(attention_mask(nd=4, ns=4, dtype=torch.int32))
