import torch

if __name__ == '__main__':
    input_ids = torch.tensor([2, 3, 5, 1])
    vocab_size = 6 #  词汇表大小
    output_size = 3 # 输出维度

    torch.manual_seed(123) # 设置随机种子
    enbbedding_layer = torch.nn.Embedding(vocab_size, output_size) # 创建embedding层
    print(enbbedding_layer.weight)

    print(enbbedding_layer(input_ids))