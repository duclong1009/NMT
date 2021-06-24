import argparse
import numpy as np
import pandas as pd
import spacy
from src.utils import seed_all, init_weights, count_parameters,epoch_time
import torchtext
from torchtext.data.utils import get_tokenizer
from src.dataset import build_vocab, data_process
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from src.trainer import train, evaluate
from src.models import Encoder, Attention, Decoder, Seq2Seq
import torch.optim as optim
import time
import math
def main(arg):
    seed_all(arg.seed)
    train_filepaths = [arg.root_path + "train.de", arg.root_path + "train.en"]
    val_filepaths = [arg.root_path + "val.de", arg.root_path + "val.en"]
    test_filepaths = [
        arg.root_path + "test_2016_flickr.de",
        arg.root_path + "test_2016_flickr.en",
    ]
    de_tokenizer = get_tokenizer("spacy", language="de")
    en_tokenizer = get_tokenizer("spacy", language="en")
    de_vocab = build_vocab(train_filepaths[0], de_tokenizer)
    en_vocab = build_vocab(train_filepaths[1], en_tokenizer)
    device = torch.device("cuda" if torch.cuda.is_availabel() else "cpu")
    train_data = data_process(train_filepaths, 64, 64)
    val_data = data_process(
        val_filepaths, de_vocab, en_vocab, de_tokenizer, en_tokenizer, 64, 64, device
    )
    test_data = data_process(
        test_filepaths, de_vocab, en_vocab, de_tokenizer, en_tokenizer, 64, 64, device
    )

    PAD_IDX = de_vocab["<pad>"]
    BOS_IDX = de_vocab["<bos>"]
    EOS_IDX = de_vocab["<eos>"]
    train_iter = DataLoader(
        train_data, batch_size=arg.batch_size, shuffle=True, num_workers=arg.num_workers
    )
    valid_iter = DataLoader(
        val_data, batch_size=arg.batch_size, shuffle=False, num_workers=arg.num_workers
    )
    test_iter = DataLoader(
        test_data, batch_size=arg.batch_size, shuffle=False, num_workers=arg.num_workers
    )
    input_dim = len(de_vocab)
    output_dim = len(en_vocab)

    enc = Encoder(
        input_dim, arg.enc_emb_dim, arg.enc_hid_dim, arg.dec_hid_dim, arg.enc_dropout
    ).to(device)
    attn = Attention(arg.enc_hid_dim, arg.dec_hid_dim, arg.attn_dim).to(device)
    dec = Decoder(
        output_dim,
        arg.dec_emb_dim,
        arg.enc_hid_dim,
        arg.dec_hid_dim,
        arg.dec_dropout,
        attn,
    ).to(device)
    model = Seq2Seq(enc, dec, device).to(device)
    model.apply(init_weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.005)
    PAD_IDX = en_vocab.stoi["<pad>"]
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    for epoch in range(arg.epochs):
        start_time = time.time()
        train_loss = train(model, train_iter, optimizer, criterion, arg.clip)
        valid_loss = evaluate(model, valid_iter, criterion)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(
            f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}"
        )
        print(
            f"\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}"
        )

    test_loss = evaluate(model, test_iter, criterion)
    print(f"| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1009)
    parser.add_argument("root_path", type=str, default="./data/en-de/")
    parser.add_argument("batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--enc_emb_dim", type=int, default=32)
    parser.add_argument("--dec_emb_dim", type=int, default=32)
    parser.add_argument("--enc_hid_dim", type=int, default=64)
    parser.add_argument("--dec_hid_dim", type=int, default=64)
    parser.add_argument("--attn_dim", type=int, default=8)
    parser.add_argument("--enc_dropout", type=float, default=0.2)
    parser.add_argument("--dec_dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--clip", type=float, default=1)
    arg = parser.parse_args()
    main(arg)
