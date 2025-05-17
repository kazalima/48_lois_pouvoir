import torch
import sys
import os
import yaml
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Ajouter le répertoire racine du projet au chemin système
project_root = '/content/48_lois_pouvoir'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Fonction pour formater l'entrée
def format_input(entry):
    instruction_text = (
        f"Ci-dessous se trouve une instruction décrivant une tâche. "
        f"Rédigez une réponse qui complète correctement la demande."
        f"\n\n### Instruction :\n{entry['instruction']}"
    )
    input_text = f"\n\n### Entrée :\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text

# Charger la configuration
with open("/content/48_lois_pouvoir/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Utiliser le GPU si disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger le modèle et le tokenizer depuis Hugging Face
model_name = config["model"]["hf_model"]
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model.eval()
print(f"Modèle chargé : {model_name}")

# Définir des questions de test
test_questions = [
    {
        "instruction": "Expliquez la loi donnée des '48 lois du pouvoir'.",
        "input": "LOI 1 : NE JAMAIS ÉCLIPSER LE MAÎTRE"
    },
    {
        "instruction": "Donnez un exemple historique illustrant la loi suivante.",
        "input": "LOI 3 : DISSIMULEZ VOS INTENTIONS"
    }
]

# Générer et afficher les réponses
for question in test_questions:
    input_text = format_input(question)
    inputs = tokenizer(input_text, return_tensors="pt", max_length=config["model"]["context_length"], truncation=True).to(device)
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=256,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text[len(input_text):].strip()
    print(f"\nQuestion : {question['instruction']} ({question['input']})")
    print(f"Réponse pré-entraînée : {response}\n")
