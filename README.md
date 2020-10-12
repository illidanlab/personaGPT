
# PersonaGPT 

### An open-domain conversational agent with many personalities

PersonaGPT follows a GPT-2 based [1](https://github.com/openai/gpt-2) transformer architecture. 
It is built from the pretrained [DialoGPT](https://github.com/microsoft/DialoGPT), which adapts the GPT-2 model to open-domain conversational responses using Reddit conversations as training corpus.
PersonaGPT is fine-tuned on the [Persona-Chat](https://arxiv.org/pdf/1801.07243) dataset, with added special tokens to better distinguish between conversational history and personality traits for dyadic conversations. 

---
## Why PersonaGPT?

The goal of PersonaGPT is to create a create an open-source database that:
1. captures the failure points of personalized conversational models, and
2. provides training data to detect fake, personalized conversational agents.

Most conversational datasets capture near-perfect human-human dialog as a source of supervision. 
However, human-chatbot conversations can exhibit failure modes that are rare in normal human-human interactions: e.g., lack of consistent personality, 

In an ongoing work, [we](https://illidanlab.github.io) are also examining cases where conversational agnets do indeed successfully "fool" human evaluators. 
In the current era of "deep fakes"(https://www.forbes.com/sites/robtoews/2020/05/25/deepfakes-are-going-to-wreak-havoc-on-society-we-are-not-prepared/#7de8ecb57494), we are trying to identify more robust *linguistic* markers (which add an additional level of difficulty) and authentication algorithms to distinguish fake conversational agents from real ones. 

Currently, [we](https://illidanlab.github.io) are collecting human-chatbot conversational data to create an active-learning style approach to improve open-domain conversational agents and to construct authentication algorithms. 
Our data-collection experiments are conducted [here](https://voigt-mckampf.xyz). 
If you are interested in participating, please read over the IRB consent information and sign-up on the website. 
You can do multiple experiments, track your data, and delete your account if you'd like to opt out at any point.

---
## Implementation Details

For reproducibility, this repository provides the following instruments to reconstruct the PersonaGPT from "scratch" (i.e., from pretrained DialoGPT or GPT-2, either of which are feasible as starting pre-trained models). 

### Requirements: ###
* Python 3.6+
* Pytorch (GPU preferred)
* [transformers](https://github.com/huggingface/transformers) package
* [tqdm](https://tqdm.github.io/)
It is highly recommended that the `pytorch` and `transformers` packages are installed under a virtual environment.

After cloning this repository, follow the directions below to set up the training environment.

### Instructions: ###
1. Go to the `.env` file and set the `save_path` to our local repository to store model checkpoints. Point `data_path` to the `/data` folder of the cloned repository. 
2. Run `preprocess_dataset.py` to preprocess `/data/train_both_original_no_cands.txt` and `/data/valid_both_original_no_cands.txt`. The original `.txt` files are obtained from the [ConvAI2 Challenge](https://github.com/DeepPavlov/convai), which may no longer be available since the ConvAI3 challenge has taken place. The ConvAI2 challenge data uses the [Persona-Chat](https://arxiv.org/pdf/1801.07243) dataset which is what is provided under the `/data` folder. 
3. Run `train.py --epochs <num epochs> --batch_size <bs> --learn_rate <lr> --grad_accum_steps <accumulation_steps>` to train the PersonaGPT model. Replace `<num epochs>`, `<bs>`, `<accumulation_steps>` and `<lr>` with the desired hyperparameters of choice. **If you have less than 12Gb GPU memory, consider using `batch size = 1`, with `4 < gradient accumulation steps <8`**.	Results, including training logs (i.e., loss per k iters) will be saved under `save_path/checkpoint`.

--- 
## References ##

1. Radford, Alec, et al. "Language models are unsupervised multitask learners." OpenAI Blog 1.8 (2019): 9.
2. Zhang, Yizhe, et al. "Dialogpt: Large-scale generative pre-training for conversational response generation." arXiv preprint arXiv:1911.00536 (2019).
3. Zhang, Saizheng, et al. "Personalizing dialogue agents: I have a dog, do you have pets too?." arXiv preprint arXiv:1801.07243 (2018).
