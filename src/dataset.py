from collections import Counter
from torchtext.vocab import Vocab
import io
import torch


def build_vocab(filepath, tokenizer):
    counter = Counter()
    with io.open(filepath, encoding="utf8") as f:
        for string_ in f:
            counter.update(tokenizer(string_))
    return Vocab(counter, specials=["<unk>", "<pad>", "<bos>", "<eos>"])


def data_process(filepaths, de_vocab, en_vocab, de_tokenizer, en_tokenizer):
    raw_de_iter = iter(io.open(filepaths[0], encoding="utf8"))
    raw_en_iter = iter(io.open(filepaths[1], encoding="utf8"))
    data = []
    for (raw_de, raw_en) in zip(raw_de_iter, raw_en_iter):
        de_tensor_ = torch.tensor(
            [de_vocab[token] for token in de_tokenizer(raw_de)], dtype=torch.long
        )
        en_tensor_ = torch.tensor(
            [en_vocab[token] for token in en_tokenizer(raw_en)], dtype=torch.long
        )
        data.append((de_tensor_, en_tensor_))
    return data


def generate_batch(data_batch):
    de_batch, en_batch = [], []
    for (de_item, en_item) in data_batch:
        de_batch.append(
            torch.cat(
                [torch.tensor([BOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0
            )
        )
        en_batch.append(
            torch.cat(
                [torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0
            )
        )
    de_batch = pad_sequence(de_batch, padding_value=PAD_IDX)
    en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
    return de_batch, en_batc
