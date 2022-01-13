import os
from transformers import (
    BigBirdConfig,
    BigBirdForMaskedLM,
    DataCollatorForWholeWordMask,
    Trainer,
    TrainingArguments,
)
import torch
from jieba_wwm_dataset import JiebaWwmDataset
from transformers import BertTokenizerFast

# from jieba_tokenizer.jieba_t import get_tokenizer

from torch.nn.parameter import Parameter
from tqdm import tqdm

dir = os.path.dirname(os.path.abspath(__file__))


def get_parnum(m):
    total = sum([param.nelement() for param in m.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))


def train():
    print("加载tokenizer")
    path = "./chinese-roberta-wwm-ext"
    config = BigBirdConfig.from_pretrained(path)
    config.architectures = "BigBirdForMaskedLM"
    config.gradient_checkpointing = True
    del config._name_or_path
    tokenizer = BertTokenizerFast.from_pretrained(path, config=config)

    print("开始读取dataset")
    path = "./chinese-roberta-wwm-ext"

    dir = os.path.dirname(os.path.abspath(__file__))
    dataset = JiebaWwmDataset(
        tokenizer=tokenizer,
        file_path=dir + "./data/merge_data_plus_shuffle_lm_512.txt",
        # file_path=dir + "/data/test.txt",
        block_size=4096,
        worker_num=16,
        tokenizer_path=path,
        load_from_disk=dir + "/data",
    )

    print("开始读取data_collator")
    data_collator = DataCollatorForWholeWordMask(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15,
    )

    model = BigBirdForMaskedLM.from_pretrained(path, config=config)

    del tokenizer.name_or_path
    del model.config._name_or_path

    init_pos_embeddings = torch.empty(3584, 768).normal_(
        mean=0.0, std=config.initializer_range
    )
    new_pos_weight = torch.cat(
        [model.bert.embeddings.position_embeddings.weight, init_pos_embeddings], axis=0
    )
    model.bert.embeddings.position_embeddings.weight = Parameter(new_pos_weight)
    model.bert.embeddings.position_embeddings.num_embeddings = model.bert.embeddings.position_embeddings.weight.shape[
        0
    ]
    model.config.max_position_embeddings = 4096
    model.bert.embeddings.token_type_ids = torch.full([1, 4096], 0)
    model.bert.embeddings.position_ids = torch.tensor([range(4096)])
    for layer_num in range(len(model.bert.encoder.layer)):
        model.bert.encoder.layer[layer_num].attention.self.max_seqlen = 4096
    get_parnum(model)

    training_args = TrainingArguments(
        output_dir="./base_BigBirdData_data_plus_gradaccu6_lm512_Checkpoint-wwm",
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=2,
        save_steps=1_000,
        save_total_limit=500,
        prediction_loss_only=True,
        gradient_accumulation_steps=50,  # 重计算，累计多个batch，更新一次参数
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
    trainer.save_model("./base_BigBirdData_data_plus_gradaccu6_lm512_Output-wwm")


if __name__ == "__main__":
    train()
