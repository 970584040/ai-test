# 多头之以利
import torch
from attention.CausalAttention import CausalAttention

class MultiHeadAttentionWrapper(torch.nn.Module):
    """
    叠加多个因果注意力
    """
    def __init__(self, d_in, d_out, content_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = torch.nn.ModuleList(
            [CausalAttention(d_in, d_out, content_length, dropout, qkv_bias=qkv_bias)
             for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

class MultiHeadAttention(torch.nn.Module):
    """
    多头注意力
    """
    def __init__(self, d_in, d_out, content_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.header_dim = d_out // num_heads
        self.W_querys = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_keys = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_values = torch.nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = torch.nn.Linear(d_out, d_out) # 线性层组合头的输出
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(content_length, content_length), diagonal=1))

    def forward(self, x):
        b, number_tokens, d_in = x.shape
        keys = self.W_keys(x)
        values = self.W_values(x)
        queries = self.W_querys(x)

        # 张量重塑（展平）
        keys = keys.view(b, number_tokens, self.num_heads, self.header_dim)
        values = values.view(b, number_tokens, self.num_heads, self.header_dim)
        queries = queries.view(b, number_tokens, self.num_heads, self.header_dim)

        # 张量转置
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        queries = queries.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3) # 计算每个头的点积
        mask_bool = self.mask.bool()[:number_tokens, :number_tokens] # 被截断为词元向量的掩码。

        attn_scores.masked_fill_(mask_bool, -torch.inf) # 填充掩码

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        content_vec = (attn_weights @ values).transpose(1, 2)

        # 组合头
        content_vec = content_vec.contiguous().view(b, number_tokens, self.d_out)

        content_vec = self.out_proj(content_vec) # 输出投影层（线性投影）
        return content_vec