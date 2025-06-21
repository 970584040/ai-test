# 简单文本分词器
import re

class TokenizerV1:
    def __init__(self,vocab):
        self.vocab = vocab
        self.int_to_str = {v: k for k, v in vocab.items()}

    def encode_tokenize(self, text):
        prec = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        prec = [item.strip() for item in prec if item.strip()]
        return [self.int_to_str[word] for word in prec ]

    def decode_tokenize(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?_!"()\'])',r'\1',text)
        return text

class TokenizerV2:
    def __init__(self,vocab):
        self.vocab = vocab
        self.int_to_str = {v: k for k, v in vocab.items()}

    def encode_tokenize(self, text):
        prec = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        prec = [item.strip() for item in prec if item.strip()]
        prec = [item if item in self.vocab else self.vocab['<|unk|>'] for item in prec] # 添加未知词元,（用）<|unk|>一环
        return [self.int_to_str[word] for word in prec ]

    def decode_tokenize(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?_!"()\'])',r'\1',text) # 删除特定标点符号前的空格
        return text