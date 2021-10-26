
# PersonaGPT 

### An open-domain conversational agent with many personalities

PersonaGPT is an open-domain conversational agent cpable of decoding _personalized_ and _controlled_ responses based on user input. It is built on the pretrained [DialoGPT-medium](https://github.com/microsoft/DialoGPT) model, following the [GPT-2](https://github.com/openai/gpt-2) architecture. 
PersonaGPT is fine-tuned on the [Persona-Chat](https://arxiv.org/pdf/1801.07243) dataset, with added special tokens to better distinguish between conversational history and personality traits for dyadic conversations. 
Furthermore, some active learning was used to train the model to do _controlled decoding_ based on certain "action codes" (e.g., talk about work, ask about music interests). 

---
## Personalized Dialog Generation 
Also called [persona modeling](https://arxiv.org/abs/1902.00098), personalized dialog generation takes as input a set of personality facts and dialog history to generate responses that are (ideally) sensible and diverse, yet consistent with past responses and the given persona profile. 

#### Example

| | Persona Facts |
|---|---|
| 1 | i am 64 years old but feel quite young. |
| 2 | i try to eat healthy but limit mcdonalds to once a week. |
| 3 | i regret working as a doctor for the last 20 years. |
| 4 | my secret hobby is making self-help youtube videos. |
| 5 | i've been to spain many times and wish i could go again. |

| Role | Response|
|---|---|
| Human | yo whats good fam? |
| Bot | hello, how are you doing? |
| Human | not bad what you doing today?|
| Bot | eating a lot healthier now that i'm in my 60s |
| Human | i'm more of a fast food guy |
| Bot | me too but i'm very healthy so i try to avoid it |
| Human | what do you do for fun? |
| Bot | i make videos for fun and eat very healthy |
| Human | nice what kind of videos? |
| Bot | i love making videos about self help stuff |
| Human | i think i need some help | 
| Bot | what kind of help would you like to get |
| Human | i'm 21 and feel lost in life | 
| Bot | i can relate, i'm only 65 |

--- 
##  Controlled Response Generation 

Another feature of PersonaGPT is generating responses that directs conversations toward certain topics of interests. The following is a self-play episode between 2 PersonaGPT agents, one interviews the other. 

| | Bot 2's Persona Facts |
|---|---|
| 1 | i am a quiet engineer. |
| 2 | i'm single and am looking for love. |
| 3 | sadly, i don't have any relatable hobbies. |
| 4 | luckily, however, i am tall and athletic. |
| 5 | on friday nights, i watch re-runs of the simpsons alone. |

| Role | Action | Response |
|---|---|---|
| Bot 1 | talk about movies | hello do you like movies? |
| Bot 2 | - | i love the simpsons. how about you? |
| Bot 1 | talk about music | i like the walking dead. do you play an instrument? |
| Bot 2 | - | yes i play the violin. |
| Bot 1 | ask about marital status | are you married or single? |
| Bot 2 | - | i am single but i am looking for love. |
| Bot 1 | talk about work | my job is working in a factory. what's your occupation? |
| Bot 2 | - | engineer. i'm very quiet so no one hears me. |
| Bot 1 | talk about hobbies | do you have any hobbies? |
| Bot 2 | - | i watch reruns of the simpsons. | 
| Bot 1 | ask about favorite food | what's your favorite food? |
| Bot 2 | - | i love pizza. how about yourself? |
| Bot 1 | ask about pets | i also love pizza. do you like animals? | 
| Bot 2 | - | i have two dogs. what is your occupation? | 
| Bot 1 | talk about work | i'm a factory worker. what's your dream job? |
| Bot 2 | - | i'd love to be a writer one day. |

--- 
## Interactive Environment

To interact with PersonaGPT, run `interact.py --mode 0` to manually enter responses and see how the bot responds. You will first be prompted to enter a set of persona facts for your partner and then engage in a short conversation. 

Use `-turns` or `--num_turns` to adjust the length of conversation (default = 8). To adjust the decoding parameters of PersonaGPT, use `--help` to bring up available settings to tune. For example, `--top_k` (default = 10) controls the number of candidate tokens at each decoding step, and `--top_p` controls the nucleus sampling. See [how to generate text from Huggingface](https://huggingface.co/blog/how-to-generate) for nuances of the GPT decoding process. 

To engage in controlled response generation, run `interact.py --mode 1`. At each turn, you will be prompted to select from a set of topics from which to steer the conversation toward. The output response by PersonaGPT may not always steer the conversation to the topic of interest, depending on the dialog history up to the current turn. 

---
## Training

Below are instructions to reconstruct the PersonaGPT from "scratch" (i.e., from pretrained DialoGPT or GPT-2, either of which are feasible as starting pre-trained models). 

### Requirements: ###
* Python 3.6+
* Pytorch (GPU preferred)
* [transformers](https://github.com/huggingface/transformers)
* [dotenv](https://pypi.org/project/python-dotenv/)
* [tqdm](https://tqdm.github.io/)
* (optional) [apex](https://www.github.com/nvidia/apex) for fp16 training
It is highly recommended that the `pytorch` and `transformers` packages are installed under a virtual environment.

After cloning this repository, follow the directions below to set up the training environment.

### Instructions: ###
1. Go to the `.env` file and set the `save_path` to your desired local repository to store model, scheduler and optimizer checkpoints. Point `data_path` to the `~/data` folder of the cloned repository. The `.env` file also contains the hyperparameter configurations:

```
epochs = 3
learn_rate = 5e-5
gradient_accumulation_steps = 64
batch_size = 1
weight_decay = 0.0
logging_steps = 10
save_steps = 250
```

Replace `epochs`, `batch_size`, `gradient_accumulation_steps` and `learn_rate` with the desired hyperparameters of choice. **Please use `batch_size = 1` and change `gradient accumulation steps`** to adjust the training batch size. This current repo version does not support parallel batching at the moment (TODO). 

2. Run `preprocess_dataset.py` to preprocess `~/data/train_both_original_no_cands.txt` and `~/data/valid_both_original_no_cands.txt`. The original `.txt` files are obtained from the [ConvAI2 Challenge](https://github.com/DeepPavlov/convai), which may no longer be available since the ConvAI3 challenge has taken place. The ConvAI2 challenge data uses the [Persona-Chat](https://arxiv.org/pdf/1801.07243) dataset which is what is provided under the `~/data` folder. 

3. Run `train.py` to train the PersonaGPT model. Results (e.g., pretrain_loss, persona_loss, ctrl_loss) will be saved under `[save_path]/samples/`. Model checkpoints are saved under `[save_path]/checkpoint/model`. 

Currently there are 2 training loops, `pretrain()` and `train_loop()`. `pretrain()` first trains model on the Persona-Chat dataset and saves the performance under `pretrain_stats`.  `train_loop()` then fine-tunes the model on active learning data, which examples of using action codes (e.g., "talk about hobbies", "ask about pets") to do controlled response generation. **The pretrained model can be used as as stand-alone dialog model for personalized dialog generation without fine-tuning on the actively learned actions.**

* `pretrain_loss`: tracks the training loss on Persona-Chat dataset during `pretrain()`.
* `persona_loss`: tracks the training loss on Persona-Chat during `train_loop()`.
* `ctrl_loss`: tracks the training loss on actively learned action codes during `train_loop()`. 

---
## Active Learning 

Currently, there are 11 possible turn-level goals that can be used for controlled response generation. 

| | Turn-level Goals | |
| --- | --- | --- |
| 1. ask about family. | 4. talk about traveling. | 7. talk about music. |
| 2. ask about pets. | 5. ask about age and gender. | 8. talk about food. |
| 3. talk about work. | 6. talk about hobbies. | 9. talk about movies. |
| 10. talk about politics. | 11. ask about marital status. |  - |

These turn-level goals are handcrafted based on the [personachat dataset](https://arxiv.org/abs/1902.00098) to cover most of the conversational directions at the turn-level. 

To actively learn **new turn-level goals**, use the [convogym repo](https://github.com/af1tang/convogym).

---

## Evaluation 

After training, an evaluation loop will run and print out a set of scores saved under `eval_stats`. 
Below is a comparison of PersonaGPT vs. other baselines on the Persona-Chat dataset using automatic evaluation metrics. Your results should look something like: 

| Model | Perplexity | F1 Score | 
|---|---|---|
| Seq2seq Baseline [3] | 29.8 | 16.2 |
| Wolf et al. [5] | 16.3 | 19.5 |
| GPT-2 baseline | 99.5 | 5.8 |
| DialoGPT baseline | 56.6 | 12.6 | 
| DialoGPT finetuned | 11.4 | 22.7 | 
| **PersonaGPT** | **10.2** | **43.4** | 

--- 

## Cite Us

Our [full paper](https://arxiv.org/abs/2110.12949v1) is now up on arXiv.

```
@misc{tang2021persona,
      title={Persona Authentication through Generative Dialogue}, 
      author={Fengyi Tang and Lifan Zeng and Fei Wang and Jiayu Zhou},
      year={2021},
      eprint={2110.12949},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

--- 

## References ##

1. Radford, Alec, et al. "Language models are unsupervised multitask learners." OpenAI Blog 1.8 (2019): 9.

2. Zhang, Yizhe, et al. "Dialogpt: Large-scale generative pre-training for conversational response generation." arXiv preprint arXiv:1911.00536 (2019).

3. Zhang, Saizheng, et al. "Personalizing dialogue agents: I have a dog, do you have pets too?." arXiv preprint arXiv:1801.07243 (2018).

4. Dinan et al., "The Second Conversational Intelligence Challenge (ConvAI2)." arXiv preprint arXiv:1902.00098 (2019).

5. Thomas Wolf et al. "Transfertransfo:  A transfer learning approach for neural network based conversational agents." arXiv preprint328arXiv:1901.08149, 2019
