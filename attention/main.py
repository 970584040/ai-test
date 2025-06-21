import torch
import torch.nn as nn
import tiktoken


def softmax_naive(x):
    """
    softmax 归一化处理
    """
    return torch.exp(x) / torch.exp(x).sum(dim=0)

if __name__ == "__main__":
    text = "Your journey starts with one step"
    
    # 1. 文本分词
    # tokens = tiktoken.get_encoding("gpt2").encode(text)
    # print(f"词元: {tokens}")
    
    # # 2. 创建词汇表
    # vocab = {word: idx for idx, word in enumerate(set(tokens))}
    # print(f"词汇表: {vocab}")
    
    # # 3. 将词元转换为索引
    # token_ids = [vocab[token] for token in tokens]
    # print(f"词元索引: {token_ids}")
    
    # # 4. 创建嵌入层并生成词向量
    # vocab_size = len(vocab)
    # embed_dim = 3  # 嵌入维度
    
    # # 设置随机种子以获得可重现的结果
    # torch.manual_seed(123)
    # embedding = nn.Embedding(vocab_size, embed_dim)
    
    # # 将索引转换为张量并获取嵌入向量
    # token_tensor = torch.tensor(token_ids)
    # inputs = embedding(token_tensor)
    
    # print(f"\n嵌入向量形状: {inputs.shape}")
    # print(f"inputs张量:\n{inputs}")
    
    # # 显示每个词对应的向量
    # print(f"\n每个词的向量表示:")
    # for i, (token, vector) in enumerate(zip(tokens, inputs)):
    #     print(f"{token:8} (x^{i+1}): {vector.detach().numpy()}")

    # 原始的手动定义张量（用于对比）
    inputs = torch.tensor([
        [0.43, 0.15, 0.89], # Your    (x^1)
        [0.55, 0.87, 0.66], # journey (x^2)
        [0.57, 0.85, 0.64], # starts  (x^3)
        [0.22, 0.58, 0.33], # with    (x^4)
        [0.77, 0.25, 0.10], # one     (x^5)
        [0.05, 0.80, 0.55]  # step    (x^6)
    ])

    # 计算点积
    query = inputs[1]
    attn_source_2 = torch.empty(inputs.shape[0])
    for i, x_i in enumerate(inputs):
        attn_source_2[i] = torch.dot(x_i, query)
    print('点积',attn_source_2)

    # 归一化
    # attn_wights_source_2_temp = attn_source_2 / torch.norm(attn_source_2)
    attn_wights_source_2_temp = attn_source_2 / attn_source_2.sum()
    print('归一化',attn_wights_source_2_temp)

    attn_wights_source_2_temp = softmax_naive(attn_source_2)
    print('softmax_naive归一化',attn_wights_source_2_temp)

    # 计算softmax(归一化)
    attn_wights_source_2 = torch.softmax(attn_source_2, dim=0)
    print('softmax归一化',attn_wights_source_2)
    
    cotenxt_vec_2 = torch.zeros(query.shape)
    for i, x_i in enumerate(inputs):
        cotenxt_vec_2 += attn_wights_source_2[i] * x_i
    print('context_vec_2',cotenxt_vec_2)

    # 计算所有的注意力权重（简化）
    # 注意力分数（点积）
    attn_source = torch.empty(6,6)
    for i, x_i in enumerate(inputs):
        for j, x_j in enumerate(inputs):
            attn_source[i,j] = torch.dot(x_i, x_j)
    print('attn_source',attn_source)
    # 以下等同于上面的
    attn_source = inputs @ inputs.T
    print('attn_source',attn_source)

    # 归一化
    attn_weights = torch.softmax(attn_source, dim=-1)
    print('attn_weights',attn_weights)

    # 注意力权重（注意力权重 * 输入）
    all_context_vec = attn_weights @ inputs
    print('all_context_vec',all_context_vec)

    x_2 = inputs[1]
    print('x_2',x_2, inputs.shape)
    # 类chart模型中，输入维度和输出维度是相同的
    d_in = inputs.shape[1] # 输入维度
    d_out = 2 # 输出维度
    # 初始化权重矩阵
    torch.manual_seed(123)
    # 初始化权重矩阵，requires_grad=False是设置为不可训练的（不在训练中更新）
    w_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False) # query是查询
    w_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False) # key是索引
    w_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False) # value是具体的值
    # 计算query, key, value
    query_2 = x_2 @ w_query
    key_2 = x_2 @ w_key
    value_2 = x_2 @ w_value
    print('query_2',query_2)
    
    keys = inputs @ w_key
    values = inputs @ w_value
    print('keys',keys)
    print('values',values)

    # 单独的自注意力分数
    key_2 = keys[1]
    attn_scores_22 = torch.dot(query_2, key_2) # 点积
    print('attn_scores_22',attn_scores_22)
    
    # 所有的自注意分数(点积)，指定query的全部自注意分数
    attn_source_2 = query_2 @ keys.T
    print(attn_source_2)

    # 计算注意力权重
    # 缩放注意力分数(权重)
    d_k = keys.shape[-1]
    attn_weights_2 = torch.softmax(attn_source_2 / d_k**0.5, dim=-1) 
    print('attn_weights_2',attn_weights_2)

    # 计算上下文向量
    context_vec_2 = attn_weights_2 @ values
    print('context_vec_2',context_vec_2)
    