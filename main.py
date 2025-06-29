import torch
from attention.CausalAttention import CausalAttention
from attention.MultiHeadAttentionWrapper import MultiHeadAttention
from module.DummyGPT import DummyGPTModel,FeedForward,ExampleDeelNeuralNetwork
import tiktoken
from module.block import TransformerBlock
from module.GPTModule import GPTModule, generate_text_simple,text_to_tokens_ids, tokens_ids_to_text

def pringt_gradients(model, x):
    """
    在模型等反向传播过程中计算梯度的函数
    """
    out = model(x)
    target = torch.tensor([0.])

    loss = torch.nn.MSELoss()
    loss = loss(out, target) # 基于目标和输出直接的差距来计算损失
    loss.backward() # 反向传播计算梯度

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name}: {param.grad.abs().mean().item():}")

def test():
    inputs = torch.tensor([
        [0.43, 0.15, 0.89],  # Your    (x^1)
        [0.55, 0.87, 0.66],  # journey (x^2)
        [0.57, 0.85, 0.64],  # starts  (x^3)
        [0.22, 0.58, 0.33],  # with    (x^4)
        [0.77, 0.25, 0.10],  # one     (x^5)
        [0.05, 0.80, 0.55]  # step    (x^6)
    ])

    torch.manual_seed(123)
    batch = torch.stack((inputs, inputs), dim=0)
    context_length = batch.shape[1]
    d_in = inputs.shape[-1]
    d_out = 2
    ca = CausalAttention(d_in, d_out, context_length, 0.0)
    content_venc = ca(batch)
    print("11context_vecs.shape:", content_venc)
    print("===================")

    batch = torch.randn(2, 6, 768)

    # 设置参数
    d_in = 768
    d_out = 768
    torch.manual_seed(123)
    context_length = batch.shape[1]
    num_heads = 12

    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, 12)
    content_venc = mha(batch)
    print("context_vecs:",content_venc)
    print("context_vecs.shape:",content_venc.shape)


    print("===================")


    GPT_CONFIG_1024M = {
        "vocab_size": 50257, # 词汇表大小
        "emb_dim": 768, # 嵌入维度
        "context_length": 1024, # 上下文长度
        "n_heads": 12, # 多头注意力头数
        "drop_rate": 0.1, # dropout丢弃率
        "n_layers": 12, # 层数
        "qkv_bias": False, # 是否使用偏置
    }

    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    text1 = "Every effort moves you"
    text2 = "Every day holds a"

    print("tokenizer.encode(text1):", tokenizer.encode(text1))
    print("tokenizer.encode(text2):", tokenizer.encode(text2))

    batch.append(torch.tensor(tokenizer.encode(text1)))
    batch.append(torch.tensor(tokenizer.encode(text2)))

    batch = torch.stack(batch, dim=0)
    print("batch.shape:", batch.shape)

    torch.manual_seed(123)
    logits = DummyGPTModel(GPT_CONFIG_1024M)(batch)
    print("logits.shape:", logits.shape)

    print("logits:", logits)

    print("=========================")
    ffn = FeedForward(cfg=GPT_CONFIG_1024M)
    x = torch.rand(2, 3, 768)
    out = ffn(x)
    print("out.shape:", out.shape)
    print("out:", out)

    print("=========================")

    x = torch.rand(2, 3, 768)
    out = ffn(x)
    print("out.shape:", out.shape)

    print("=================")

    layer_size=[3, 3, 3, 3, 3, 1]  # 添加输出层维度
    sample_input = torch.tensor([1, 0., -1.])
    torch.manual_seed(123)
    model_with_shortcut = ExampleDeelNeuralNetwork(layer_size=layer_size, use_shortcut=True)
    pringt_gradients(model_with_shortcut, sample_input)


    print("====================")
    torch.manual_seed(123)
    x = torch.rand(2, 4, 768)
    block = TransformerBlock(GPT_CONFIG_1024M)
    output = block(x)
    print("output.shape", output.shape)

    print("====================")
    torch.manual_seed(123)
    x = torch.rand(2, 4, 768)
    block = TransformerBlock(GPT_CONFIG_1024M)
    output = block(x)
    print("output.shape", output.shape)

    print("====================")
    torch.manual_seed(123)
    model = GPTModule(GPT_CONFIG_1024M)
    output = model(batch)
    print("output.shape", output.shape)

    total_params = sum(p.numel() for p in model.parameters()) # 计算模型参数总数
    print(f"Total parameters: {total_params}")

    # 计算内存需求
    total_memory = total_params * 4 / 1024 / 1024 # 每个参数4字节，转换为MB
    print(f"Total memory: {total_memory} MB")

    # 因为GPTModule使用了权重共享（将词元嵌入层作为输出层重复使用），所以实际参数数量为：
    total_params = (total_params - sum(p.numel() for p in model.out_head.parameters()))
    print(f"Total parameters: {total_params}")

    print("=========生成文本===========")
    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    print("encoded:", encoded)

    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    # model = GPTModule(GPT_CONFIG_1024M)
    model.eval() # 关闭 dropout
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor, 
        max_new_tokens=6, 
        context_size=GPT_CONFIG_1024M["context_length"]
    )
    print("out:", out)
    print("out.shape:", len(out[0]), out.shape)

    print("====================")

def gpt_main():
    GPT_CONFIG_1024M = {
        "vocab_size": 50257, # 词汇表大小
        "emb_dim": 768, # 嵌入维度
        "context_length": 1024, # 上下文长度
        "n_heads": 12, # 多头注意力头数
        "drop_rate": 0.1, # dropout丢弃率
        "n_layers": 12, # 层数
        "qkv_bias": False, # 是否使用偏置
    }
    model = GPTModule(GPT_CONFIG_1024M)

    start_context = "Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")
    tokens_ids = generate_text_simple(
        model=model,
        idx=text_to_tokens_ids(start_context, tokenizer),
        max_new_tokens=10,
        context_size=GPT_CONFIG_1024M["context_length"]
    )

    print("tokens_ids:", tokens_ids)
    print("tokens_ids output text:", tokens_ids_to_text(tokens_ids, tokenizer))

if __name__ == '__main__':
    # test()
    gpt_main()
