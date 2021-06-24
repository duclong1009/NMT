import math
import time
from tqdm.auto import tqdm
import torch.nn as nn
import torch.optim as optim
import torch

def train(model: nn.Module,
          iterator: torch.utils.data.DataLoader,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          clip: float):

    model.train()
    epoch_loss = 0

    for src, trg in tqdm(iterator):
        optimizer.zero_grad()
        src = src.permute(1,0)
        output = model(src, 64)
        output = output[1:].view(-1, output.shape[-1])
        trg = trg.permute(1,0)
        trg = trg[1:].reshape(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate(model: nn.Module,
             iterator: torch.utils.data.DataLoader,
             criterion: nn.Module):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for _, (src, trg) in enumerate(iterator):
            src = src.permute(1,0)
            output = model(src, 64, 0) #turn off teacher forcing
            output = output[1:].view(-1, output.shape[-1])
            trg = trg.permute(1,0)
            trg = trg[1:].reshape(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def epoch_time(start_time: int,
               end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

