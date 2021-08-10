# bigbird中文预训练模型

## 简介
+ [BigBird](https://github.com/google-research/bigbird)是由google与2020年[发表](https://arxiv.org/abs/2007.14062)的模型结构，利用稀疏注意力机制，把二次依赖关系降为线性依赖，因此更适合做长文本任务。
+ 本项目利用[transformers](https://github.com/huggingface/transformers)库，[CLUE开源语料](https://github.com/brightmart/nlp_chinese_corpus)，预训练中文版本`BigBird`模型，并借助[huggingface社区](https://huggingface.co/models)分享，如果对您有用，欢迎`star`项目。

## 预训练步骤
1. 提取文本长度在512-1024的文本
2. jieba分词，按照词频创建字典
3. 预训练

## 使用
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

model = BigBirdModel.from_pretrained('Lowin/chinese-bigbird-tiny')
tokenizer = JiebaTokenizer.from_pretrained('Lowin/chinese-bigbird-tiny')
```
## 模型训练参数
|模型|参数量|block|vocab_size|layers|hidden_size|heads|显存|batch_size|gradient_accumulation_steps|total_step|
|-|-|-|-|-|-|-|-|-|-|-|
|chinese-bigbird-tiny|10.8M|1024|20_000|4|220|11|8000*2|16|4|150K
|chinese-bigbird-small|41.36M|1024|30_000|6|512|8|12000|18|6|150K

## 参考

https://github.com/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb
