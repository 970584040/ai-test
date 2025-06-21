import re

import tiktoken
import GPTDatasetv1
from torch.utils.data import DataLoader


def main():
    with open("./the-verdict.txt", "r", encoding="utf-8") as f:
        text = f.read()
    print(len(text))
    # 文本分割
    prec = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    prec = [item.strip() for item in prec if item.strip()]
    print(len(prec))
    # 获取唯一词元列表,并且按照字母排序
    all_words = sorted(list(set(prec)))
    all_words.extend(["<|endoftext|>", "<|unk|>"])
    # 词汇表
    vocab = {word: i for i, word in enumerate(all_words)}

    print(vocab.items())

    for i, word in enumerate(vocab.items()):
        print(f"{i}: {word}")
        if i >= 50:
            break

def tiktoken_test():
    """
    BPE 分词器
    :return:
    """
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    text = ("Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace.")
    integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print(integers)

    text = ("Akwirw ier")
    integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print(integers)
    for i in integers:
        print(i,tokenizer.decode([i]))

    print(tokenizer.decode(integers))

def create_dataloader_v1(txt, batch_size=4, max_length=512, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetv1.GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader

def gpt_DatasetV1(txt):
    data_loader = create_dataloader_v1(txt,  batch_size=2, max_length=10, stride=5, shuffle=False)
    data_iter = iter(data_loader)
    first_batch = next(data_iter)
    print(first_batch)


if __name__ == "__main__":
    with open("./the-verdict.txt", "r", encoding="utf-8") as f:
        text = f.read()

    gpt_DatasetV1(text)