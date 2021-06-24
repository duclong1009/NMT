from collections import Counter
from torchtext.vocab import Vocab
import io


def build_vocab(filepath, tokenizer):
    counter = Counter()
    with io.open(filepath, encoding="utf8") as f:
        for string_ in f:
            counter.update(tokenizer(string_))
    return Vocab(counter, specials=["<unk>", "<pad>", "<bos>", "<eos>"])


def data_process(
    filepaths, de_vocab, en_vocab, de_tokenizer, en_tokenizer, len_de, len_en, device
):
    de_pad = de_vocab["<pad>"]
    en_pad = en_vocab["<pad>"]

    raw_de_iter = iter(io.open(filepaths[0], encoding="utf8"))
    raw_en_iter = iter(io.open(filepaths[1], encoding="utf8"))
    data = []

    for (raw_de, raw_en) in zip(raw_de_iter, raw_en_iter):
        de_tensor_ = torch.tensor(
            [de_vocab["<bos>"]]
            + [de_vocab[token] for token in de_tokenizer(raw_de)]
            + [de_vocab["<eos>"]],
            dtype=torch.long,
        )

        en_tensor_ = torch.tensor(
            [en_vocab["<bos>"]]
            + [en_vocab[token] for token in en_tokenizer(raw_en)]
            + [en_vocab["<eos>"]],
            dtype=torch.long,
        )

    temp_de = de_tensor_.shape[0]
    add_de = len_de - temp_de
    add_de = de_pad * torch.ones(add_de).long()
    temp_en = en_tensor_.shape[0]
    add_en = len_en - temp_en
    add_en = en_pad * torch.ones(add_en).long()
    de_tensor_ = torch.cat((de_tensor_, add_de), dim=0).to(device)
    en_tensor_ = torch.cat((en_tensor_, add_en), dim=0).to(device)
    data.append((de_tensor_, en_tensor_))
    return data
