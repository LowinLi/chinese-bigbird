"""
@author: LowinLi
tokenizer
"""

import jieba_fast
from transformers import BertTokenizer
import os

jieba_fast.enable_parallel()


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


def get_tokenizer(pretrained_dir=os.path.dirname(os.path.abspath(__file__))):
    return JiebaTokenizer.from_pretrained(pretrained_dir)


if __name__ == "__main__":
    tokenizer = get_tokenizer()
    out = tokenizer.encode("我爱北京天安门")
    print(out)
    print(len(out))
