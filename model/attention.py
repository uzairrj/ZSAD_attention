from torch.nn import Module, Dropout, Linear, LayerNorm
import torch.nn.functional as F
import torch

class CrossAttention(Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads

        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.q_proj = Linear(dim, dim)
        self.k_proj = Linear(dim, dim)
        self.v_proj = Linear(dim, dim)
        self.out_proj = Linear(dim, dim)

        self.dropout = Dropout(dropout)
        self.norm = LayerNorm(dim)

    def _qkv(self, x, context):
        batch_size, seq_len_x, _ = x.size()
        _, seq_len_ctx, _ = context.size()

        q = self.q_proj(x).view(batch_size, seq_len_x, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(batch_size, seq_len_ctx, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(batch_size, seq_len_ctx, self.num_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def forward(self, x, context):
        batch_size, seq_len_x, _ = x.size()

        q, k, v = self._qkv(x, context)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len_x, self.dim)
        output = self.out_proj(attn_output)
        
        return self.norm(output + x)
