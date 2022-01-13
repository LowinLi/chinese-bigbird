import jieba_fast
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import os
import pandas as pd
import time
from multiprocessing import Process, Manager, Lock, managers


class JiebaWwmDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        file_path,
        block_size,
        worker_num,
        tokenizer_path,
        load_from_disk=None,
    ):
        self._file_path = file_path
        self._worker_num = worker_num
        self._tokenizer = tokenizer
        self._block_size = block_size
        self._tokenizer_path = tokenizer_path
        self._total_data = 0
        self.examples = []
        if load_from_disk:
            # self._multi_load(load_from_disk)
            self._single_load(load_from_disk)
        else:
            self._multi_jieba()

    def _single_load(self, load_from_disk):
        files = os.listdir(load_from_disk)
        files = sorted(files)[:9]
        for file in tqdm(files):
            if "reftmp" in file:
                df = pd.read_csv(os.path.join(load_from_disk, file), sep=",")
                df.columns = ["input_ids", "chinese_refs"]
                df = df.dropna(how="any")
                for record in df.to_dict(orient="records"):
                    self.examples.append(
                        {
                            "input_ids": [
                                int(x)
                                for x in record["input_ids"][: self._block_size]
                                if x.strip()
                            ],
                            "chinese_ref": [
                                int(x)
                                for x in record["chinese_refs"]
                                if x.strip() and int(x) < self._block_size
                            ],
                        }
                    )
                    self._total_data += 1

    def _multi_load(self, load_from_disk):
        # global manager
        lock = Lock()

        with Manager() as manager:
            return_dict = manager.dict()  # 空字典
            i = 0
            for file in os.listdir(load_from_disk):
                process_list = []
                if "reftmp" in file:
                    p = Process(
                        target=worker_load_disk,
                        args=(os.path.join(load_from_disk, file), i, return_dict, lock),
                    )
                    i += 1
                    p.daemon = False
                    p.start()
                    process_list.append(p)
            for p in process_list:
                p.join()
            for k, v in return_dict.items():
                self.examples.extend(v)

    def _multi_jieba(self):
        with open(self._file_path, "r") as f:
            lines = f.readlines()
            self._total_data = len(lines) - 1
            process_list = []
            batch_size = int(self._total_data / self._worker_num) + 1
            for i in range(self._worker_num):
                sub_lines = lines[i * batch_size : (i + 1) * batch_size]
                p = Process(
                    target=worker_jieba_process,
                    args=(sub_lines, i, self._tokenizer_path),
                )
                p.start()
                process_list.append(p)

            for i in process_list:
                p.join()
            print("分词结束")

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return self._total_data


def worker_load_disk(file, num, return_dict, lock):
    print(file)
    examples = []
    df = pd.read_csv(file, sep=",")
    df.columns = ["input_ids", "chinese_refs"]
    df = df.dropna(how="any")
    for record in df.to_dict(orient="records"):
        examples.append(
            {
                "input_ids": [int(x) for x in record["input_ids"] if x.strip()],
                "chinese_ref": [int(x) for x in record["chinese_refs"] if x.strip()],
            }
        )
    lock.acquire()
    return_dict[str(num)] = examples
    lock.release()


def worker_jieba_process(lines, num, tokenizer_file):
    from transformers import BertTokenizerFast

    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_file)
    with open(f"data/reftmp_{num}.tsv", "w") as wf:
        wf.write("input_ids,chinese_refs\n")
        for line in lines:
            input_ids, chinese_refs, _ = prepare_ref_jieba(line, tokenizer)
            write_line = (
                " ".join([str(x) for x in input_ids])
                + ","
                + " ".join([str(x) for x in chinese_refs])
            )
            wf.write(write_line + "\n")


def prepare_ref_jieba(text, tokenizer, max_length=4096):
    input_ids = tokenizer(
        text, add_special_tokens=True, truncation=True, max_length=max_length
    )["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[:max_length])
    line = ""
    token_char_indexs = []
    token_char_index = 0
    token_char_dict = {}
    for token_index, token in enumerate(tokens):
        token_char_dict[token_char_index] = token_index
        line += token
        token_char_index += len(token)
        token_char_indexs.append(token_char_index)
    word_char_index = 0
    word_char_indexs = []
    words = jieba_fast.lcut(line)
    for word in words:
        word_char_index += len(word)
        word_char_indexs.append(word_char_index)
    ref_indexs = [
        token_char_dict[x] for x in token_char_indexs if x not in word_char_indexs
    ]
    return (input_ids[:max_length], [x for x in ref_indexs if x < max_length], words)


if __name__ == "__main__":
    from transformers import BertTokenizerFast

    path = "./chinese-roberta-wwm-ext"
    tokenizer = BertTokenizerFast.from_pretrained(path)
    # result = prepare_ref_jieba(["我爱北京天安门"], tokenizer)
    # print(result)

    dir = os.path.dirname(os.path.abspath(__file__))
    dataset = JiebaWwmDataset(
        tokenizer=tokenizer,
        file_path=dir + "/data/merge_data_plus_shuffle_lm_512.txt",
        # file_path=dir + "/data/test.txt",
        block_size=4096,
        worker_num=16,
        tokenizer_path=path,
        # load_from_disk=dir + "/data",
    )
