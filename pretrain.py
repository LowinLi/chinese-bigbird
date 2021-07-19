"""
@author: LowinLi
预训练
"""

import os
from transformers import (
    BigBirdConfig,
    BigBirdForMaskedLM,
    LineByLineTextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
import torch
from torch.utils.data.dataset import Dataset, Iterable
from jieba_tokenizer.jieba_t import get_tokenizer
from pathlib import Path
import linecache

dir = os.path.dirname(os.path.abspath(__file__))

class LazyTextDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size):
        self._filename = file_path
        self._tokenizer = tokenizer
        self._block_size = block_size
        self._total_data = 0
        with open(file_path, "r") as f:
            self._total_data = len(f.readlines()) - 1

    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        encoding = self._tokenizer(line, add_special_tokens=True, truncation=True, max_length=self._block_size)["input_ids"]
        example = {"input_ids": torch.tensor(encoding, dtype=torch.long)}
        return example

    def __len__(self):
        return self._total_data

def get_parnum(m):
    total = sum([param.nelement() for param in m.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))


def train():
    config = BigBirdConfig(
        vocab_size=20_000,
        hidden_size=220,
        num_attention_heads=11,
        num_hidden_layers=4,
        max_position_embeddings=1024,
        gradient_checkpointing=True,
    )
    model = BigBirdForMaskedLM(config=config)
    get_parnum(model)
    tokenizer = get_tokenizer()
    print("开始读取dataset")
    dataset = LazyTextDataset(
        tokenizer=tokenizer,
        file_path=dir + "/merge_data_plus_shuffle_lm_512.txt",
        # file_path=dir + "/data/wiki_zh/AA/wiki_00",
        block_size=1024,
    )
    print("开始读取data_collator")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15,
    )

    training_args = TrainingArguments(
        output_dir="./BigBirdData_data_plus_gradaccu6_lm512_Checkpoint",
        overwrite_output_dir=True,
        num_train_epochs=40,
        per_device_train_batch_size=16,
        save_steps=10_000,
        save_total_limit=500,
        prediction_loss_only=True,
        gradient_accumulation_steps=4, #重计算，累计多个batch，更新一次参数
        max_grad_norm=1
        # group_by_length=True, #相似长度sample放在一起
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        
    )
    print("开始训练")
    trainer.train()
    trainer.save_model("./BigBirdData_data_plus_gradaccu6_lm512_Output")


if __name__ == "__main__":
    train()
