
# PersonaGPT 

### An open-domain conversational agent with many personalities

PersonaGPT follows a [GPT-2 based](https://github.com/openai/gpt-2) transformer architecture. 
It is built from the pretrained [DialoGPT](https://github.com/microsoft/DialoGPT), which adapts the GPT-2 model to open-domain conversational responses using Reddit conversations as training corpus.
PersonaGPT is fine-tuned on the [Persona-Chat](https://arxiv.org/pdf/1801.07243) dataset, with added special tokens to better distinguish between conversational history and personality traits for dyadic conversations. 

---
## Why PersonaGPT?

The goal of PersonaGPT is to create a create an open-source database that:
1. captures the failure points of personalized conversational models, and
2. provides training data to detect fake, personalized conversational agents.

Most conversational datasets capture near-perfect human-human dialog as a source of supervision. 
However, human-chatbot conversations can exhibit failure modes that are rare in normal human-human interactions: e.g., lack of consistent personality, 

In an ongoing work, [we](https://illidanlab.github.io) are also examining cases where conversational agents do indeed successfully "fool" human evaluators. 
In the current era of ["deep fakes"](https://www.forbes.com/sites/robtoews/2020/05/25/deepfakes-are-going-to-wreak-havoc-on-society-we-are-not-prepared/#7de8ecb57494), we are trying to identify more robust *linguistic* markers (which add an additional level of difficulty) and authentication algorithms to distinguish fake conversational agents from real ones. 

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
* [dotenv](https://pypi.org/project/python-dotenv/)
* [tqdm](https://tqdm.github.io/)
* (optional) [apex](https://www.github.com/nvidia/apex) for fp16 training
It is highly recommended that the `pytorch` and `transformers` packages are installed under a virtual environment.

After cloning this repository, follow the directions below to set up the training environment.

### Instructions: ###
1. Go to the `.env` file and set the `save_path` to our local repository to store model checkpoints. Point `data_path` to the `/data` folder of the cloned repository. The `.env` file also contains the hyperparameter configurations:

```
epochs = 3
learn_rate = 5e-5
gradient_accumulation_steps = 8
batch_size = 1
weight_decay = 0.0
logging_steps = 10
save_steps = 250
```

Replace `epochs`, `batch_size`, `gradient_accumulation_steps` and `learn_rate` with the desired hyperparameters of choice. **If you have less than 12Gb GPU memory, consider using `batch size = 1`, with `gradient accumulation steps` between [4-8]**.

2. Run `preprocess_dataset.py` to preprocess `/data/train_both_original_no_cands.txt` and `/data/valid_both_original_no_cands.txt`. The original `.txt` files are obtained from the [ConvAI2 Challenge](https://github.com/DeepPavlov/convai), which may no longer be available since the ConvAI3 challenge has taken place. The ConvAI2 challenge data uses the [Persona-Chat](https://arxiv.org/pdf/1801.07243) dataset which is what is provided under the `/data` folder. 

3. Run `train.py` to train the PersonaGPT model. Results (i.e., loss per `<logging_steps>`) will be saved under `save_path/samples/`. Model checkpoints are saved under `save_path/checkpoint/model`.

--- 
## References ##

1. Radford, Alec, et al. "Language models are unsupervised multitask learners." OpenAI Blog 1.8 (2019): 9.

2. Zhang, Yizhe, et al. "Dialogpt: Large-scale generative pre-training for conversational response generation." arXiv preprint arXiv:1911.00536 (2019).

3. Zhang, Saizheng, et al. "Personalizing dialogue agents: I have a dog, do you have pets too?." arXiv preprint arXiv:1801.07243 (2018).
