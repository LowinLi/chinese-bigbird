# bigbird中文预训练模型

## 简介
+ [BigBird](https://github.com/google-research/bigbird)是由google与2020年[发表](https://arxiv.org/abs/2007.14062)的模型结构，利用稀疏注意力机制，把二次依赖关系降为线性依赖，因此更适合做长文本任务。
+ 本项目利用[transformers](https://github.com/huggingface/transformers)库，[CLUE开源语料](https://github.com/brightmart/nlp_chinese_corpus)，预训练中文版本`BigBird`模型，并借助[huggingface社区](https://huggingface.co/Lowin)分享，如果对您有用，欢迎`star`项目。

## 预训练步骤
1. 提取文本长度在512-1024的文本(tiny、mini、small);提取文本长度在512-4096的文本(base、wwm-base)
2. jieba分词，按照词频创建字典(tiny、mini、small、base);用chinese-roberta-wwm-ext原版字典(wwm-base)
3. 掩词预训练(tiny、mini、small、base)；WWM掩字预训练(wwm-base)

## 使用
+ tiny、mini、small、base
```python
import jieba_fast
from transformers import BertTokenizer
from transformers import BigBirdModel

class JiebaTokenizer(BertTokenizer):
    def __init__(
        self, pre_tokenizer=lambda x: jieba_fast.cut(x, HMM=False), *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.pre_tokenizer = pre_tokenizer

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for text in self.pre_tokenizer(text):
            if text in self.vocab:
                split_tokens.append(text)
            else:
                split_tokens.extend(super()._tokenize(text))
        return split_tokens

model = BigBirdModel.from_pretrained('Lowin/chinese-bigbird-tiny-1024')
tokenizer = JiebaTokenizer.from_pretrained('Lowin/chinese-bigbird-tiny-1024')
```

+ wwm-base

```python
from transformers import BertTokenizer
from transformers import BigBirdModel

model = BigBirdModel.from_pretrained('Lowin/chinese-bigbird-wwm-base-4096')
tokenizer = BertTokenizer.from_pretrained('Lowin/chinese-bigbird-wwm-base-4096')
```

## 模型训练参数
| 模型                          | 粒度              | 参数量 | max_length | vocab_size | layers | hidden_size | heads | total_step |
|-------------------------------|-------------------|--------|------------|------------|--------|-------------|-------|------------|
| chinese-bigbird-tiny-1024     | jieba分词与字结合 | 10.8M  | 1024       | 20_000     | 4      | 220         | 11    | 150K       |
| chinese-bigbird-mini-1024     | jieba分词与字结合 | 18.9M  | 1024       | 25_000     | 5      | 300         | 10    | 150K       |
| chinese-bigbird-small-1024    | jieba分词与字结合 | 41.36M | 1024       | 30_000     | 6      | 512         | 8     | 150K       |
| chinese-bigbird-base-4096     | jieba分词与字结合 | 119.5M | 4096       | 40_000     | 12     | 768         | 12    | 30K        |
| chinese-bigbird-wwm-base-4096 | 字                | 105M   | 4096       | 21_128     | 12     | 768         | 12    | 2K         |
## 参考

https://github.com/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb
