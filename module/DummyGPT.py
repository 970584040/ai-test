import torch
import torch.nn as nn

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg

        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.position_embedding = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.dropout_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_ln = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.token_embedding(in_idx)
        pos_embeds = self.position_embedding(torch.arange(seq_len, device=in_idx.device))

        x = tok_embeds + pos_embeds
        x = self.dropout_emb(x)
        x = self.trf_blocks(x)
        x = self.final_ln(x)
        logits = self.out_head(x)

        return logits
        
class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
    
    def forward(self, x):
        return x
    
class LayerNorm(nn.Module):
    """
    层归一化类
    """
    def __init__(self, ebm_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(ebm_dim))
        self.shift = nn.Parameter(torch.zeros(ebm_dim))

    def forward(self, x):
        mean = x.mean(dim=-1,keepdim=True)
        var = x.var(dim=-1,keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
    
    def forward(self, x):
        return x
    
class GELU(nn.Module):
    """
    GELU激活函数类
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) *
                                          (x + 0.044715 * torch.pow(x, 3))))
    
class FeedForward(nn.Module):
    """
    前馈神经网络类
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], cfg["emb_dim"] * 4),
            # nn.GELU(),
            GELU(),
            nn.Linear(cfg["emb_dim"] * 4, cfg["emb_dim"])
        )
    
    def forward(self, x):
        return self.layers(x)
    
class ExampleDeelNeuralNetwork(nn.Module):
    """
    演示快捷连接（跳跃连接、残差连接）
    """
    def __init__(self, layer_size, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.Sequential(
            nn.Sequential(nn.Linear(layer_size[0], layer_size[1]), GELU()),
            nn.Sequential(nn.Linear(layer_size[1], layer_size[2]), GELU()),
            nn.Sequential(nn.Linear(layer_size[2], layer_size[3]), GELU()),
            nn.Sequential(nn.Linear(layer_size[3], layer_size[4]), GELU()),
            nn.Sequential(nn.Linear(layer_size[4], layer_size[5]), GELU()),
        )
       
    def forward(self, x):
       for layer in self.layers:
           layer_outpt = layer(x)
           if self.use_shortcut and x.shape == layer_outpt.shape:
               x = x + layer_outpt
           else:
               x = layer_outpt
       return x
    
if __name__ == "__main__":
    torch.manual_seed(123)
    batch_example = torch.randn(2, 5)
    ln = LayerNorm(ebm_dim=5)
    out_ln = ln(batch_example)
    mean = out_ln.mean(dim=-1,keepdim=True)
    var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
    print("mean:", mean)
    print("var:", var)
    print("out_ln:", out_ln)
    print("=========================")
    

