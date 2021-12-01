# bigbird中文预训练模型

## 简介
+ [BigBird](https://github.com/google-research/bigbird)是由google与2020年[发表](https://arxiv.org/abs/2007.14062)的模型结构，利用稀疏注意力机制，把二次依赖关系降为线性依赖，因此更适合做长文本任务。
+ 本项目使用[transformers](https://github.com/huggingface/transformers)库，[CLUE开源语料](https://github.com/brightmart/nlp_chinese_corpus)，预训练中文版本`BigBird`模型，并借助[huggingface社区](https://huggingface.co/Lowin)分享，如果对您有用，欢迎`star`项目。

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

模型详见[Huggingface Model Hub](https://huggingface.co/Lowin)

## 模型训练参数
| 模型                          | 粒度              | 参数量 | max_length | vocab_size | layers | hidden_size | heads | total_step |
|-------------------------------|-------------------|--------|------------|------------|--------|-------------|-------|------------|
| chinese-bigbird-tiny-1024     | jieba分词与字结合 | 10.8M  | 1024       | 20_000     | 4      | 220         | 11    | 150K       |
| chinese-bigbird-mini-1024     | jieba分词与字结合 | 18.9M  | 1024       | 25_000     | 5      | 300         | 10    | 150K       |
| chinese-bigbird-small-1024    | jieba分词与字结合 | 41.36M | 1024       | 30_000     | 6      | 512         | 8     | 150K       |
| chinese-bigbird-base-4096     | jieba分词与字结合 | 119.5M | 4096       | 40_000     | 12     | 768         | 12    | 30K        |
| chinese-bigbird-wwm-base-4096 | 字                | 105M   | 4096       | 21_128     | 12     | 768         | 12    | 2K（在chinesee-roberta-wwm-ext权重基础继续WWM预训练）         |

## 下游finetune效果
### clue分类任务
#### finetune参数
| 参数       | 参数值 |
|------------|--------|
| 学习率     | 5e-05  |
| batch_size | 50     |
| epochs     | 2      |

#### acc
| 模型                                            | TNEWS(字符长度平均37) | IFLYTEK(字符长度98%分位712) |
|-------------------------------------------------|-----------------------|-----------------------------|
| chinese-roberta-wwm-ext(baseline-截断前512字符) | 0.566                 | 0.59                        |
| chinese-bigbird-tiny-1024                       | 0.551                 | 0.47                        |
| chinese-bigbird-mini-1024                       | 0.548                 | 0.501                       |
| chinese-bigbird-small-1024                      | 0.559                 | 0.565                       |
| chinese-bigbird-base-4096                       | 0.557                 | 0.589                       |
| chinese-bigbird-wwm-base-4096                   | 0.568                 | 0.573                       |

#### 推断平均用时（ms）

+ onnx量化单核cpu推荐平均用时（ms）
+ Intel(R) Core(TM) i7-4790 CPU @ 3.60GHz

| 模型                                            | TNEWS(字符长度平均37) | IFLYTEK(字符长度98%分位712) |
|-------------------------------------------------|-----------------------|-----------------------------|
| chinese-roberta-wwm-ext(baseline-截断前512字符) | 47                    | 577                         |
| chinese-bigbird-tiny-1024                       | 4                     | 42                          |
| chinese-bigbird-mini-1024                       | 6                     | 63                          |
| chinese-bigbird-small-1024                      | 11                    | 118                         |
| chinese-bigbird-base-4096                       | 34                    | 367                         |
| chinese-bigbird-wwm-base-4096                   | 47                    | 600                         |


## 参考

https://github.com/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb
