#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 20:26:03 2020

@author: af1tang
"""
import torch, argparse
from load_configs import model, tokenizer, opts, device, p1_tok, p2_tok, act_tok, start_tok
from utils import *

action_space = [ 'ask about kids.', "ask about pets.", 'talk about work.', 
               'ask about marital status.', 'talk about travel.', 'ask about age and gender.',
        'ask about hobbies.', 'ask about favorite food.', 'talk about movies.', 
        'talk about music.', 'talk about politics.']


def generate_next(bot_input_ids, do_sample=True, top_k=10, top_p=.92,
                  max_length=1000, pad_token=tokenizer.eos_token_id):
    full_msg = model.generate(bot_input_ids, do_sample=True,
                                              top_k=top_k, top_p=top_p, 
                                              max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    msg = to_data(full_msg.detach()[0])[bot_input_ids.shape[-1]:]
    return msg

## Interactive Mode w/ User ##
def get_personas():
    # custom personas for conversation
    personas = []
    for i in range(5):
        response = input(">> Fact %d: "%(i+1))+ tokenizer.eos_token
        personas.append(response)
    personas = tokenizer.encode(''.join(['<|p2|>'] + personas + ['<|sep|>'] + ['<|start|>']))
    return personas

def interact(choice, personas, length=8, top_k=10, top_p=.92, max_length=1000):
    dialog_hx = []

    # chat time
    for step in range(length):
        if choice ==1:
            # encode the user input
            user_inp = tokenizer.encode(input(">> User: ") + tokenizer.eos_token)
            # append to the chat history
            dialog_hx.append(user_inp)
                
            # generated a response while limiting the total chat history to 1000 tokens, 
            bot_input_ids = to_var([personas + flatten(dialog_hx)]).long()
            msg = generate_next(bot_input_ids, top_k=top_k, top_p=top_p, max_length=max_length)
            dialog_hx.append(msg)
            print("Bot: {}".format(tokenizer.decode(msg, skip_special_tokens=True)))

        else:
            act = None
            while act not in action_space:
                display_dialog_history(dialog_hx)
                print()
                print(" actions: ")
                for k,v in enumerate(action_space): print(k,v)
                try:
                    act = action_space[int(input(" input [0-10]: " ))]
                except:
                    act = None
            print()
            action_prefix = tokenizer.encode(''.join(['<|act|> '] + [act] + ['<|p1|>'] + [] + ['<|sep|>'] + ['<|start|>']))
            bot_input_ids = to_var([action_prefix + flatten(dialog_hx)]).long()
            
            # generate query conditioned on action
            msg = generate_next(bot_input_ids, top_k=top_k, top_p=top_p, max_length=max_length)
            dialog_hx.append(msg)
            
            # generate bot response
            bot_input_ids = to_var([personas+ flatten(dialog_hx)]).long()
            msg = generate_next(bot_input_ids, top_k=top_k, top_p=top_p, max_length=max_length)
            dialog_hx.append(msg)
    if choice == 2:
        display_dialog_history(dialog_hx)
    return dialog_hx

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Conversational parameters for interacting with personaGPT.')
    parser.add_argument('-M', '--mode', type=int, 
                        dest='mode', default=1,
                        help='''mode (0 or 1) of interaction: 
                        (0) user gives prompts to persona model,
                        (1) user picks action codes for controlled decoding.''')
    parser.add_argument('-turns', '--num_turns', type=int, 
                        dest='turns', default=8,
                        help='number of turns in conversation (default 8)')
    parser.add_argument('-maxlen', '--max_length', type=int, 
                        dest='max_length', default=1000,
                        help='max num of tokens in convo (default 1000)') 
    parser.add_argument('-k', '--top_k', type=int,
                        dest='top_k', default=10,
                        help='top_k sampling parameter (default 10)')
    parser.add_argument('-p', '--top_p', type=float, 
                        dest='top_p', default=.92,
                        help='nucleus sampling parameter (default 0.92)')    

    args = parser.parse_args()
    personas = get_personas()
    dialog_hx = interact(args.mode, personas, length=args.turns, 
             top_k=args.top_k, top_p=args.top_p,
             max_length=args.max_length)