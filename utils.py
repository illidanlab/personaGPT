#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 14:17:28 2020

@author: af1tang
"""
import torch, os, pickle, matplotlib.pyplot as plt
import torch.nn as nn, torch.nn.functional as F
from itertools import groupby
from load_configs import tokenizer, p1_tok, p2_tok, start_tok, opts, device, create_dir

## Utils ##
flatten = lambda l: [item for sublist in l for item in sublist]
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))
    
def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()

def to_var(x):
    if not torch.is_tensor(x):
        x = torch.Tensor(x)
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def process_conv(row, tokenizer, eos = True, make_flat=True):
    if eos:
        conv = list([tokenizer.encode(x) + [tokenizer.eos_token_id] for x in row])
    else: conv = list([tokenizer.encode(x) for x in row])
    if make_flat: conv = flatten(conv)
    return conv

def split_by_index(seq, sep):
    result = []
    for el in seq:
        result.append(el)
        if el == sep:
            yield result
            result = []
            
def filter_turn_indices(x):
    filtered = [[t[1] for t in list(g)] for k,g in groupby(list(enumerate(x)), lambda x: x[1]==tokenizer.eos_token_id) if not k]
    return filtered

def display_dialog_history(dialog_hx):
    for j, line in enumerate(dialog_hx):
        msg = tokenizer.decode(line)
        if j %2 == 0:
            print(">> User: "+ msg)
        else:
            print("Bot: "+msg)
            print()

### plotting ###    
def plot_losses(stats, title='loss'):
    create_dir(opts.plot_path)
    x = list(sorted(stats.keys()))
    loss = [stats[i][title] for i in x]
    plt.plot(x, loss, label= title)
    plt.legend()
    plt.title("%s" %title)
    plt.tight_layout()
    plt.savefig(os.path.join(opts.plot_path,'%s.png'%title))
    plt.close()