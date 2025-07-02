# 简化因果注意力""
import torch

class CausalAttention(torch.nn.Module):
    def __init__(self, d_in, d_out, content_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(content_length, content_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        values = self.W_value(x)
        querys = self.W_query(x)

        attn_sources = querys @ keys.transpose(1, 2) # 将维度1和2转置，将维度保持在第一个位置（0）
        # 使用-1e9代替-inf防止数值不稳定
        attn_sources.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -1e9)
        attn_weights = torch.softmax(attn_sources / (keys.shape[-1]**0.5), dim=-1)
        attn_weights = self.dropout(attn_weights) # 对注意力权重进行随机置零，防止过拟合。

        context_vec = attn_weights @ values
        return context_vec