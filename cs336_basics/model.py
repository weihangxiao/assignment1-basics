"""Transformer language model implementation with custom components."""

import torch.nn as nn
import torch
import math
from einops import rearrange, einsum
from cs336_basics.nn_utils import softmax
import einx

class Linear(nn.Module):
    """Linear layer with truncated normal initialization.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        device: Device for parameters
        dtype: Data type for parameters
    """
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        weight = torch.empty((out_features, in_features), device=device, dtype=dtype)
        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(weight, mean=0, std=std, a=-3*std, b=3*std)
        
        self.weight = nn.Parameter(weight)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply linear transformation.
        
        Args:
            x: Input tensor (batch, seq_len, in_features)
            
        Returns:
            Output tensor (batch, seq_len, out_features)
        """
        out = torch.einsum("b t d, o d -> b t o", x, self.weight)
        return out
        


class Embedding(nn.Module):
    """Token embedding layer.
    
    Args:
        num_embeddings: Vocabulary size
        embedding_dim: Embedding dimension
        device: Device for parameters
        dtype: Data type for parameters
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        weight = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        std = math.sqrt(2 / (num_embeddings + embedding_dim))
        nn.init.trunc_normal_(weight, mean=0, std=std, a=-3*std, b=3*std)
        
        
        self.weight = nn.Parameter(weight)
        
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up embeddings.
        
        Args:
            token_ids: Token indices (batch, seq_len)
            
        Returns:
            Embeddings (batch, seq_len, embedding_dim)
        """
        return self.weight[token_ids]
    


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    Args:
        d_model: Model dimension
        eps: Epsilon for numerical stability
        device: Device for parameters
        dtype: Data type for parameters
    """
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized tensor
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        result = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        
        return result.to(in_dtype)

def silu(x:torch.Tensor) -> torch.Tensor:
    """SiLU activation: x * sigmoid(x)."""
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    """SwiGLU feedforward network.
    
    Args:
        d_model: Model dimension
        d_ff: Feedforward dimension
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU: W2(SiLU(W1(x)) * W3(x))."""
        return self.w2(silu(self.w1(x)) * self.w3(x))


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE).
    
    Args:
        theta: Base frequency
        d_k: Key dimension
        max_seq_len: Maximum sequence length
        device: Device for buffers
    """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        theta = torch.tensor(theta, device=device)
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        
        self.register_buffer(
            'cos_sin', 
            precompute_freqs_cis(d_k, max_seq_len, theta),
            persistent=False
        )
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """Apply rotary embeddings.
        
        Args:
            x: Input tensor (..., seq_len, dim)
            token_positions: Position indices
            
        Returns:
            Tensor with rotary embeddings applied
        """
        # [..., seq_len, dim], [seq_len,] -> [..., seq_len, dim]
        cos_sin = self.cos_sin[:x.size(-2)] if token_positions is None else self.cos_sin[token_positions]
        return apply_rotary_emb(x, cos_sin)


def precompute_freqs_cis(head_dim: int, max_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute rotary embedding frequencies.
    
    Args:
        head_dim: Attention head dimension
        max_len: Maximum sequence length
        theta: Base frequency
        
    Returns:
        Concatenated cos and sin values (max_len, head_dim)
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim)) # shape (head_dim/2,)
    t = torch.arange(max_len, device=freqs.device).float() # shape (max_len,)
    freqs = torch.outer(t, freqs) # equal to einsum('i,j->ij', t, freqs), shape (max_len, head_dim/2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64, equal to torch.complex(torch.cos(freqs), torch.sin(freqs))
    cos_sin = torch.cat([freqs_cis.real, freqs_cis.imag], dim=-1) # [cos, sin] shape (max_len, head_dim/2 * 2)
    return cos_sin

def apply_rotary_emb(x:torch.Tensor, cos_sin:torch.Tensor):
    """Apply precomputed rotary embeddings to tensor.
    
    Args:
        x: Input tensor
        cos_sin: Precomputed cos and sin values
        
    Returns:
        Rotated tensor
    """
    # x1, x2 = torch.chunk(x, 2, dim=-1)
    x1, x2 = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    cos, sin = torch.chunk(cos_sin, 2, dim=-1)
    x_out = torch.stack([x1 * cos - x2 * sin, 
                         x1 * sin + x2 * cos], dim=-1)
    return x_out.reshape(*x.shape).type_as(x)

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """Compute scaled dot-product attention.
    
    Args:
        Q: Query tensor (..., queries, d_k)
        K: Key tensor (..., keys, d_k)
        V: Value tensor (..., keys, d_v)
        mask: Optional attention mask
        
    Returns:
        Attention output (..., queries, d_v)
    """
    attention_score = einsum(Q, K, " ... queries d_k, ... keys d_k -> ... queries keys")
    attention_score = attention_score / (Q.shape[-1] ** 0.5)
    if mask is not None:
        attention_score.masked_fill_(mask == 0, float("-inf"))
        
    attention_weights = softmax(attention_score, dim=-1)
    
    value_v = einsum(attention_weights, V, "... queries keys, ... keys d_v -> ... queries d_v")
    return value_v

class MultiheadSelfAttention(nn.Module):
    """Multi-head self-attention with causal masking.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        pos_encode: Optional rotary positional encoding
        theta: Optional theta for rotary embeddings
    """
    def __init__(self, d_model: int, num_heads: int, pos_encode: RotaryPositionalEmbedding | None = None, theta: float | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_proj = Linear(d_model, self.d_k * self.num_heads)
        self.k_proj = Linear(d_model, self.d_k * self.num_heads)
        self.v_proj = Linear(d_model, self.d_k * self.num_heads)
        self.output_proj = Linear(self.d_k * num_heads, d_model)
        self.pos_encode = pos_encode
        self.theta = theta
    
    def forward(self, x, token_positions: torch.Tensor | None = None):
        """Apply multi-head self-attention.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            token_positions: Optional position indices
            
        Returns:
            Attention output (batch, seq_len, d_model)
        """
        *b, num_seq, d_model = x.shape
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(*b, num_seq, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(*b, num_seq, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(*b, num_seq, self.num_heads, self.d_k).transpose(1, 2)
        if token_positions is None:
            token_positions = einx.rearrange("seq -> b... seq", torch.arange(num_seq, device=x.device), b=[1] * len(b))
        token_positions = rearrange(token_positions, "... seq -> ... 1 seq")
        if self.theta is not None:
            Q = self.pos_encode(Q, token_positions)
            K = self.pos_encode(K, token_positions)
        
        causal_mask = torch.tril(torch.ones(num_seq, num_seq, device=x.device))
        causal_mask = causal_mask.view(1, 1, num_seq, num_seq)
        
        value_v = scaled_dot_product_attention(Q, K, V, causal_mask)
        
        value_v = value_v.transpose(1, 2).contiguous().view(*b, num_seq, -1)
        return self.output_proj(value_v)


class TransformerBlock(nn.Module):
    """Transformer block with attention and feedforward layers.
    
    Args:
        d_model: Model dimension
        d_ff: Feedforward dimension
        num_heads: Number of attention heads
        pos_encode: Optional rotary positional encoding
        theta: Optional theta for rotary embeddings
    """
    def __init__(self, d_model: int, d_ff: int, num_heads: int, pos_encode: RotaryPositionalEmbedding | None = None, theta: float | None = None):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.sa = MultiheadSelfAttention(d_model, num_heads, pos_encode, theta)
        self.ff = SwiGLU(d_model, d_ff)
    
    
    def forward(self, x):
        """Apply transformer block with residual connections.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        x = self.sa(self.ln1(x)) + x
        
        x = self.ff(self.ln2(x)) + x
        
        ### Post-Normalization ###
        # x = self.ln1(self.sa(x) + x)
        # x = self.ln2(self.ff(x) + x)
        return x


class Transformer_LM(nn.Module):
    """Transformer language model.
    
    Args:
        vocab_size: Vocabulary size
        context_length: Maximum context length
        d_model: Model dimension
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        d_ff: Feedforward dimension
        pos_encode: Optional rotary positional encoding
        theta: Optional theta for rotary embeddings
    """
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, pos_encode: RotaryPositionalEmbedding | None = None, theta: float | None = None):
        super().__init__()
        self.token_embedding = Embedding(vocab_size, d_model)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model, d_ff, num_heads, pos_encode, theta) for _ in range(num_layers)])
        self.norm = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
        
    
    def forward(self, x):
        """Forward pass through language model.
        
        Args:
            x: Token IDs (batch, seq_len)
            
        Returns:
            Logits (batch, seq_len, vocab_size) without softmax
        """
        x = self.token_embedding(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.norm(x)
        x = self.lm_head(x)
        return x # no softmax
        