from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def load_hf_model(config, device):
    model_name = config["model"]["hf_model"]
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, model_name
