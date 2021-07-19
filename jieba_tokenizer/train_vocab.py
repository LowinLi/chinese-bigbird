"""
@author: LowinLi
字典处理
"""

import jieba_fast
from tqdm import tqdm
import pandas as pd
import re

jieba_fast.enable_parallel()


def main(file="../data/0609/merge_data_plus_shuffle_lm_512.txt"):
    token_dict = {}
    with open(file, "r") as f:
        lines = f.readlines()
    for line in tqdm(lines):
        tokens = jieba_fast.lcut(line)
        for token in tokens:
            if token in token_dict:
                token_dict[token] += 1
            else:
                token_dict[token] = 1
    df = pd.DataFrame({"word": token_dict.keys(), "count": token_dict.values()})
    pat = re.compile(
        "[\u4e00-\u9fa5]|[\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b]"
    )  # 所有中文和中文标点
    df["chinese"] = df["word"].apply(lambda x: bool(pat.search(x)))
    df = df[df["chinese"]]
    df = df.sort_values("count", ascending=False)
    with open("special_tokens.txt", "r") as f:
        vocabs = f.read()

    df["length"] = df["word"].apply(lambda x: len(x))
    # 5000个字和标点
    for word in (df["word"][df["length"] == 1].iloc[:5000]).tolist():
        vocabs += word + "\n"
    # 15000-106个词
    for word in (df["word"][df["length"] > 1].iloc[: 15000 - 106]).tolist(): 
        vocabs += word + "\n"
    with open("../data/0609/vocab.txt", "w") as f:
        f.write(vocabs)


if __name__ == "__main__":
    main()
