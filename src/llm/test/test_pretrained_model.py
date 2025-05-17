import yaml
import tiktoken
import torch
from src.llm.dataset import format_input
from src.llm.model import load_model
from src.llm.utils.previous_chapters import generate, text_to_token_ids, token_ids_to_text

def test_pretrained_model():
    # Charger la configuration
    with open("/content/48_lois_pouvoir/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Initialiser le tokenizer et le device
    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Charger le modèle pré-entraîné (avant fine-tuning)
    model, model_name = load_model(config, device)
    print(f"Modèle pré-entraîné chargé : {model_name}")

    # Définir quelques questions de test
    test_questions = [
        {
            "instruction": "Expliquez la loi donnée des '48 lois du pouvoir'.",
            "input": "LOI 1 : NE JAMAIS ÉCLIPSER LE MAÎTRE"
        },
        {
            "instruction": "Donnez un exemple historique illustrant la loi suivante.",
            "input": "LOI 3 : DISSIMULEZ VOS INTENTIONS"
        },
        {
            "instruction": "Comment appliquer la loi suivante dans un contexte professionnel moderne ?",
            "input": "LOI 7 : FAITES TRAVAILLER LES AUTRES POUR VOUS, MAIS APPROPRIEZ-VOUS LES LAURIERS"
        }
    ]

    # Générer et afficher les réponses
    for question in test_questions:
        input_text = format_input(question)
        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=config["model"]["context_length"],
            eos_id=50256
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response = generated_text[len(input_text):].replace("### Réponse :", "").strip()
        print(f"\nQuestion : {question['instruction']} ({question['input']})")
        print(f"Réponse pré-entraînée : {response}\n")

if __name__ == "__main__":
    test_pretrained_model()
