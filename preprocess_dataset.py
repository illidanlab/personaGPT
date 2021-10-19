#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 11:40:29 2020

@author: af1tang
"""
from tqdm import tqdm
import torch, os, pickle
import torch.nn as nn, torch.nn.functional as F
import numpy as np, random
import pandas as pd

from load_configs import model, tokenizer, opts, device, data_path, save_path
from utils import *

## Persona Preprocess ##
def preprocess_convai(filename):
    raw_data = open(filename).read().strip().split('\n')
    data, count = {}, 0
    curr_convo, curr_ps, curr_pt = [], [], []
    indices = []
    
    person_a = 'your persona'
    person_b = "partner's persona"
    with tqdm(total = len(raw_data)) as pbar:
        turn_count, ctx_count = 1,0 #init cycle
        for idx, line in enumerate(raw_data):
            if person_a in line[0:20]:
                if (turn_count != 0) and (len(curr_ps)>1 and len(curr_pt)>1 and len(curr_convo)>1):
                    if idx > 1:
                        if curr_convo[0] == '__SILENCE__' :
                            p1 = curr_ps; p2 = curr_pt; curr_convo = curr_convo[1:]
                        else:
                            p1 = curr_pt; p2 = curr_ps
                        data[count] = { 'inp': process_conv([curr_convo[0]], tokenizer),
                                        'labels': process_conv(curr_convo[1:],tokenizer), #to_data(torch.cat(curr_convo,dim=-1)[0]), 
                                       'p_src': process_conv(p1, tokenizer,make_flat=False), #to_data(torch.cat(curr_ps,dim=-1)[0]),
                                       'p_trg': process_conv(p2, tokenizer, make_flat=False)}#to_data(torch.cat(curr_pt,dim=-1)[0])}
                        count+=1
                    curr_convo, curr_ps, curr_pt = [], [], []
                    turn_count=0

                words = line.split()
                turn_id, words = int(words[0]), ' '.join(words[3:])
                curr_ps.append(words)

                ctx_count +=1
                assert ctx_count == turn_id
                
            elif person_b in line[0:20]:
                if (turn_count != 0) and (len(curr_ps)>1 and len(curr_pt)>1 and len(curr_convo)>1):
                    if idx > 1:
                        if curr_convo[0] == '__SILENCE__' :
                            p1 = curr_ps; p2 = curr_pt; curr_convo = curr_convo[1:]
                        else:
                            p1 = curr_pt; p2 = curr_ps
                        data[count] = { 'inp': process_conv([curr_convo[0]], tokenizer),
                                        'labels': process_conv(curr_convo[1:],tokenizer), #to_data(torch.cat(curr_convo,dim=-1)[0]), 
                                       'p_src': process_conv(p1, tokenizer,make_flat=False), #to_data(torch.cat(curr_ps,dim=-1)[0]),
                                       'p_trg': process_conv(p2, tokenizer, make_flat=False)}#to_data(torch.cat(curr_pt,dim=-1)[0])}
                        count+=1
                    curr_convo, curr_ps, curr_pt = [], [], []
                    turn_count=0
                words = line.split()
                turn_id, words = int(words[0]), ' '.join(words[3:])
                curr_pt.append(words)

                ctx_count +=1
                assert ctx_count == turn_id

                
            else:
                if ctx_count !=0:
                    turn_count = ctx_count *1 
                    ctx_count =0
                    indices.append(idx)
                        
                src_line, trg_line = line.split('\t')
                src_words = src_line.split()
                src_idx, src_line = src_words[0], ' '.join(src_words[1:])

                curr_convo.append(src_line) 
                curr_convo.append(trg_line)#turn)
                
                turn_count +=1
                assert turn_count == int(src_idx)
                
            pbar.update(1)
        
    return data

def convert_to_XY(old_data):
    data = []
    print("building training set...")
    for i in range(len(old_data)):
        p1 = [tokenizer.decode(p) for p in old_data[i]['p_src']]
        p2 = [tokenizer.decode(p) for p in old_data[i]['p_trg']]

        convo = old_data[i]['inp'] + old_data[i]['labels']
        dialog_hx = list(split_by_index(convo,tokenizer.eos_token_id))
        #if len(dialog_hx) < 30:
        dialog_hx = [line + [tokenizer.eos_token_id] for line in dialog_hx[:20]] # limit by max len of convo
        p1_ctx = tokenizer.encode(''.join(['<|p1|>'] + p1 + ['<|sep|>'] + ['<|start|>']))
        p2_ctx = tokenizer.encode(''.join(['<|p2|>'] + p2 + ['<|sep|>'] + ['<|start|>']))
        for t in range(len(dialog_hx)):
            x = dialog_hx[:t]
            y = dialog_hx[t]
            if t == 0:
                x = p1_ctx[:-1] 
                y = [p1_ctx[-1]] + y
            elif t %2 ==0:
                x = p1_ctx + flatten(x)
            else:
                x = p2_ctx + flatten(x)
            data.append((x,y))
    return data

def build_active_data():
    df = pd.read_csv(os.path.join(data_path, 'active_learning_data.csv'))
    X, y = df['context'].tolist(), df['response'].tolist()
    X, y = [tokenizer.encode(x) for x in X], [tokenizer.encode(yy) for yy in y]
    data = {'X':X, 'y':y}
    return data
    
    
if __name__ == '__main__':        
    train_data = preprocess_convai(os.path.join(data_path, 'train_both_original_no_cands.txt'))
    train_data = convert_to_XY(train_data)
    val_data = preprocess_convai(os.path.join(data_path, 'valid_both_original_no_cands.txt'))
    val_data = convert_to_XY(val_data)
    active_data = build_active_data()
    with open(opts.raw_data_path, 'wb') as f: pickle.dump(train_data, f)
    with open(opts.val_data_path, 'wb') as f: pickle.dump(val_data, f)
    with open(opts.active_data_path, 'wb') as f: pickle.dump(active_data, f)