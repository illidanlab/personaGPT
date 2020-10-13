#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 20:26:03 2020

@author: af1tang
"""
import torch
from load_configs import model, tokenizer, opts, device, p1_tok, p2_tok, start_tok
from utils import *

## Interactive Mode w/ User ##
def interact(length=8, top_k=10, top_p=.92, max_length=1000, use_persona=True):
    p1_tensor, p2_tensor, start_tensor = torch.tensor([[p1_tok]]), torch.tensor([[p2_tok]]), torch.tensor([[start_tok]])
    sep_tok = torch.tensor(tokenizer.sep_token_id).view(1,1)
    if use_persona:
        #cls_tok = torch.tensor(tokenizer.cls_token_id).view(1,1)
        default_persona = [tokenizer.encode('Not available.'+tokenizer.eos_token, return_tensors='pt')] 
        personas = []
        for i in range(5):
            response = tokenizer.encode(input(">> Fact %d: "%(i+1))+ tokenizer.eos_token, return_tensors='pt')
            personas.append(response)
        personas = [p1_tensor] + [tokenizer.encode("person 1: ", return_tensors='pt')] + \
                    default_persona + [sep_tok] + [p2_tensor] + \
                    [tokenizer.encode("person 2: ", return_tensors='pt')] + personas + [sep_tok] + [start_tensor]

        chat_history_ids = to_var(torch.cat(personas, dim=-1))

    else:
        chat_history_ids = to_var(start_tensor)
    for step in range(length):
        # encode the user input
        new_user_input_ids = tokenizer.encode(input(">> User: ") + tokenizer.eos_token, return_tensors='pt').to(device)
    
        # append to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)# if ((use_persona) or (step > 0)) else new_user_input_ids
        
        # generated a response while limiting the total chat history to 1000 tokens, 
        chat_history_ids = model.generate(bot_input_ids, do_sample=True, 
                                          top_k=top_k, top_p=top_p, 
                                          max_length=max_length, pad_token_id=tokenizer.eos_token_id)
        # pretty print last ouput tokens from bot
        print("Bot: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))


if __name__=="__main__":
    interact(use_persona=True)