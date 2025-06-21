# 自注意力类
import torch.nn as nn
import torch


class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        self.W_query = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
        self.W_key = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
        self.W_value = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

    def forward(self, x):
        queries = x @ self.W_query
        keys = x @ self.W_key
        values = x @ self.W_value

        attn_source = queries @ keys.T
        attn_weights = torch.softmax(attn_source / (keys.shape[-1] ** 0.5), dim=-1)
        context_vec = attn_weights @ values
        return context_vec
        

class SelfAttention_v2(nn.Module):
    """
    使用线性层实现自注意力
    """
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attn_source = queries @ keys.T
        attn_weights = torch.softmax(attn_source / (keys.shape[-1] ** 0.5), dim=-1)
        context_vec = attn_weights @ values
        return context_vec