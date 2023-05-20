from typing import List
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from ..arguments import DataArguments


def load_text_data(data_args):
    corpus_path = data_args.data_path
    data = []
    with open(corpus_path) as f:
        for l in f:
            data.append(l.strip().split('\t')[1])
    return data


def split_train_valid(data: List[str], validation_percentage):
    num_valid = int(len(data) * validation_percentage / 100.0)
    train = data[:-num_valid]
    valid = data[-num_valid:]
    return train, valid


class PretrainDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, data: List[str], data_args: DataArguments):
        super(PretrainDataset, self).__init__()
        self.tokenizer = tokenizer
        self.corpus_path = data_args.data_path
        self.data = data
        self.data_args = data_args
        self.max_len = data_args.max_len

    def create_one_example(self, text: str):
        return self.tokenizer(text,
                              truncation='only_first',
                              max_length=self.data_args.max_len,
                              padding=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.create_one_example(self.data[item])
