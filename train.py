#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 14:10:21 2020

@author: af1tang
"""
import torch, os, pickle
import numpy as np
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup

from load_configs import model, tokenizer, stats, opts, device, create_dir, p1_tok, p2_tok, start_tok
from utils import *

## model saving ##
def checkpoint(model, tokenizer, optimizer, scheduler, stats):
    create_dir(opts.output_dir)
    model.save_pretrained(opts.output_dir)
    tokenizer.save_pretrained(opts.output_dir)
    torch.save(opts, os.path.join(opts.output_dir, "training_opts.bin"))
    torch.save(optimizer.state_dict(), os.path.join(opts.output_dir, 'optimizer.pt'))
    torch.save(scheduler.state_dict(), os.path.join(opts.output_dir, 'scheduler.pt'))
    with open(os.path.join(opts.output_dir, 'stats.pkl'), 'wb') as f: pickle.dump(stats,f)
    
## Training Pipeline ##
def train_loop(data, model, tokenizer, stats=None): 
    inps = [[start_tok]  for i in range(len(data))]#[data[i]['inp'] for i in range(len(data))]
    convos = [data[i]['inp'] + data[i]['labels'] for i in range(len(data))]

    P1 = [ [p1_tok] + tokenizer.encode("person 1: ") + flatten(data[i]['p_src']) + [tokenizer.sep_token_id] for i in range(len(data))]
    P2 = [  [p2_tok] + tokenizer.encode("person 2: ") + flatten(data[i]['p_trg']) + [tokenizer.sep_token_id] for i in range(len(data))]
    
    databunch = list(zip(inps,convos, P1, P2))
    
    train_dataset = ConvDataset(databunch)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=opts.batch_size, 
                                  collate_fn=collate_tuple, drop_last=True)
    del inps, convos, P1, P2, databunch
    
    # calculate total steps
    if opts.max_steps > 0:
        t_total = opts.max_steps
        opts.num_train_epochs = opts.max_steps // (len(train_dataloader) // opts.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // opts.gradient_accumulation_steps * opts.num_train_epochs

    ## set up optimizers and schedulers ##
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [ {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                                         "weight_decay": opts.weight_decay},
                                    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                                     "weight_decay": 0.0} ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=opts.lr, eps=opts.eps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=opts.warmup_steps, 
                                                num_training_steps=t_total)
    # loading optimizer settings
    if (opts.model_name_or_path and os.path.isfile(os.path.join(opts.model_name_or_path, "optimizer.pt"))
                                and os.path.isfile(os.path.join(opts.model_name_or_path, "scheduler.pt")) ):
        # load optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(opts.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(opts.model_name_or_path, "scheduler.pt")))
    if opts.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=opts.fp16_opt_level)
    if stats is not None:
        global_step = len(stats)
        epochs_trained = global_step // (len(train_dataloader) // opts.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // opts.gradient_accumulation_steps)
        print("Resuming Training ... ")
    else:
        print("New Training ... ")
        stats = {}
        global_step, epochs_trained, steps_trained_in_current_epoch = 0,0,0
    ## start training ##
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    print("Re-sizing model ... ")
    model.resize_token_embeddings(len(tokenizer))
    for epoch in range(epochs_trained, opts.num_train_epochs):
        data_iter = iter(train_dataloader)
        for step in range(len(train_dataloader)):
            # skip previous steps if re-training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            # process batch
            batch = data_iter.next()
            if opts.use_token_ids:
                inp, labels, p1, p2, m_x, m_y, m_p1, m_p2, px, py, pp1, pp2, tx, ty, tp1, tp2 = map(to_var,batch ); del batch
            else:
                inp, labels, p1, p2, m_x, m_y, m_p1, m_p2, px, py, pp1, pp2 = map(to_var,batch ); del batch
                tx, ty, tp1, tp2 = None,None,None,None
            ctx = torch.cat([p1,p2, inp], dim=-1); del p1,p2,inp # context = p1 | p2 | x
            p_ctx = torch.cat([pp1,pp2, px], dim=-1); del pp1,pp2,px # positional id's of context
            m_ctx = torch.cat([m_p1, m_p2, m_x], dim=-1); del m_p1,m_p2, m_x # attention masks of context
            m_full = torch.cat([m_ctx, m_y], dim=-1)    # concat masks into 1 attention mask
            if tx is not None:
                t_ctx = torch.cat([tp1, tp2, tx], dim=-1); del tp1, tp2, tx # token id's of context
            # forward pass #
            model.train()
            # forward through history (obtain "past" contextual states)
            # (k,v) x bs x heads x t x dim
            if opts.use_token_ids:
                _, past = model(ctx, attention_mask=m_ctx, position_ids = p_ctx, token_type_ids = t_ctx); del _, m_ctx
                outp = model(labels, attention_mask=m_full, position_ids = py, token_type_ids = ty,
                             past=past, labels=labels)
            else:
                _, past = model(ctx, attention_mask=m_ctx, position_ids = p_ctx); del _, m_ctx
                outp = model(labels, attention_mask=m_full, position_ids = py, 
                             past=past, labels=labels)
            # forward through prediction

            # (loss), lm_logits, presents, (all hidden_states), (attentions)
            loss = outp[0]; del outp
            if opts.gradient_accumulation_steps > 1:
                loss = loss / opts.gradient_accumulation_steps
            # backward
            if opts.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            tr_loss += loss.item()
            
            # gradient accumulation
            if (step+1) % opts.gradient_accumulation_steps == 0:
                if opts.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), opts.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), opts.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                
                # reporting 
                if global_step % opts.logging_steps ==0:
                    stats[global_step] = {'loss': (tr_loss - logging_loss) / opts.logging_steps, 
                                          'lr': scheduler.get_last_lr()[0]}
                    logging_loss = tr_loss
                    
                    print('Epoch: %d | Iter: %d | loss: %.3f | lr: %s ' %( 
                    epoch, global_step, stats[global_step]['loss'], str(stats[global_step]['lr'])) )
                    
                if global_step % opts.save_steps==0:
                    print("Saving stuff ... ")
                    checkpoint(model, tokenizer, optimizer, scheduler, stats)
                    plot_losses(stats, title='loss' )
                    plot_losses(stats, title='lr')
                    print("Done.")
    return stats

def evaluate_loop(data, model, tokenizer, mode='PM-GPT'):
    # prepare validation dataloader
    if mode == 'PM-GPT':
        inps = [[start_tok]  for i in range(len(data))] #[data[i]['inp'] for i in range(len(data))]
        convos = [data[i]['inp'] + data[i]['labels'] for i in range(len(data))]
        P1 = [ [p1_tok] + tokenizer.encode("person 1: ") + flatten(data[i]['p_src']) + [tokenizer.sep_token_id] for i in range(len(data))]
        P2 = [  [p2_tok] + tokenizer.encode("person 2: ") + flatten(data[i]['p_trg']) + [tokenizer.sep_token_id] for i in range(len(data))]
    else:
        inps = [data[i]['inp'] for i in range(len(data))]
        convos = [data[i]['labels'] for i in range(len(data))]
        P1 = [ tokenizer.encode("person 1: ") + flatten(data[i]['p_src']) for i in range(len(data))]
        P2 = [  tokenizer.encode("person 2: ") + flatten(data[i]['p_trg']) for i in range(len(data))]
    
    databunch = list(zip(inps,convos, P1, P2))
    
    val_dataset = ConvDataset(databunch)
    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=opts.batch_size, 
                                  collate_fn=collate_tuple, drop_last=True)
    del inps, convos, P1, P2, data        
    data_iter = iter(val_dataloader)

    # evaluation loop
    print("Validating ... ")
    with torch.no_grad():
        eval_stats, total_steps, val_loss, val_f1_score = {}, 0, 0.0, 0.0
        model.resize_token_embeddings(len(tokenizer))
        model.eval()
        for step in range(len(val_dataloader)):
            # process batch
            batch = data_iter.next()
            if opts.use_token_ids:
                inp, labels, p1, p2, m_x, m_y, m_p1, m_p2, px, py, pp1, pp2, tx, ty, tp1, tp2 = map(to_var,batch ); del batch
            else:
                inp, labels, p1, p2, m_x, m_y, m_p1, m_p2, px, py, pp1, pp2 = map(to_var,batch ); del batch
                tx, ty, tp1, tp2 = None,None,None,None
            ctx = torch.cat([p1,p2, inp], dim=-1) # context = p1 | p2 | x
            p_ctx = torch.cat([pp1,pp2, px], dim=-1) # positional id's of context
            m_ctx = torch.cat([m_p1, m_p2, m_x], dim=-1) # attention masks of context
            m_full = torch.cat([m_ctx, m_y], dim=-1)    # concat masks into 1 attention mask
            if tx is not None:
                t_ctx = torch.cat([tp1, tp2, tx], dim=-1) # token id's of context
            # forward pass #
            if opts.use_token_ids:
                _, past = model(ctx, attention_mask=m_ctx, position_ids = p_ctx, token_type_ids = t_ctx); del _, m_ctx
                outp = model(labels, attention_mask=m_full, position_ids = py, token_type_ids = ty,
                             past=past, labels=labels)
            elif mode != 'PM-GPT':
                _, past = model(inp)
                outp = model(labels, past=past, labels=labels)
            else:
                _, past = model(ctx, attention_mask=m_ctx, position_ids = p_ctx); del _, m_ctx
                outp = model(labels, attention_mask=m_full, position_ids = py, 
                             past=past, labels=labels)
            # forward through prediction
            loss = outp[0]
            # f1 score calc
            #ytrue =  to_data(labels[..., 1:].contiguous().view(-1))
            ytrue=np.array( filter_turn_indices(to_data(labels[...,1:].contiguous().view(-1)) ) )
            #ypred = to_data(outp[1][..., :,:].contiguous().topk(1)[1].view(-1))
            ypred=np.array( filter_turn_indices(to_data( outp[1][..., :-1, :].contiguous().topk(1)[1].view(-1)) ) ) 
            min_len = min(len(ypred), len(ytrue))
            hits = [set(ypred[i]).intersection(set(ytrue[i])) for i in range(min_len)]#set(ytrue).intersection(set(ypred))
            prec = [len(hits[i])/len(ypred[i]) for i in range(min_len)]
            rec = [len(hits[i])/len(ytrue[i]) for i in range(min_len)]
            f1 = np.mean([2*(prec[i]*rec[i])/(prec[i] + rec[i]+1e-3) for i in range(min_len)])
            val_f1_score += f1
            val_loss += loss.mean().item()
            total_steps +=1 
            if total_steps%20 ==0: print(total_steps)
            
    val_loss = val_loss / total_steps 
    val_f1_score = val_f1_score / total_steps
    perplexity = torch.exp(torch.tensor(val_loss)).item()
    eval_stats = {'perplexity': perplexity, 'loss': val_loss, 'f1': val_f1_score}
    print("Done.")
    return eval_stats

if __name__ == '__main__':        
    with open(opts.raw_data_path, 'rb') as f: data = pickle.load(f)
    stats = train_loop(data, model, tokenizer, stats)

    with open(opts.val_data_path, 'rb') as f: data = pickle.load(f)
    eval_stats = evaluate_loop(data, model, tokenizer)
       