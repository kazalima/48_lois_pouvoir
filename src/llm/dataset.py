from torch.utils.data import Dataset
import torch
from functools import partial
from transformers import AutoTokenizer

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.encoded_texts = []

        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Réponse :\n{entry['output']}"
            full_text = instruction_plus_input + response_text

            tokens = tokenizer(
                full_text,
                max_length=1024,
                truncation=True,
                padding=False,
                return_tensors=None,
            )["input_ids"]

            self.encoded_texts.append(tokens)

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)

def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device="cpu"):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

def format_input(entry):
    instruction_text = (
        f"Ci-dessous se trouve une instruction décrivant une tâche. "
        f"Rédigez une réponse qui complète correctement la demande."
        f"\n\n### Instruction :\n{entry['instruction']}"
    )
    input_text = f"\n\n### Entrée :\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text
