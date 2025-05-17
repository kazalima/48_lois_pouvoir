import json
import torch
from torch.utils.data import DataLoader
import yaml
from src.llm.dataset import InstructionDataset, custom_collate_fn
from src.llm.model_hf import load_hf_model
from src.llm.trainer_hf import compute_loss, train_model, plot_losses
from src.llm.inference_hf import generate_responses
from functools import partial

def main():
    # Charger la configuration
    with open("/content/48_lois_pouvoir/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Définir le device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # Charger les données depuis le fichier JSONL
    data = []
    # Change loading from json.load to reading line by line for JSONL format
    with open(config["data"]["processed"], "r", encoding="utf-8") as f:
        for line in f:
            # Check if the line is not empty before trying to parse
            if line.strip():
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line: {line.strip()}")
                    print(f"Details of error: {e}")
                    continue # Skip this line and continue with the next

    train_portion = int(len(data) * 0.85)  # 85% pour l'entraînement
    test_portion = int(len(data) * 0.1)    # 10% pour le test
    val_portion = len(data) - train_portion - test_portion  # 5% pour la validation
    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]
    print(f"Taille de l'ensemble d'entraînement : {len(train_data)}")
    print(f"Taille de l'ensemble de validation : {len(val_data)}")
    print(f"Taille de l'ensemble de test : {len(test_data)}")

    # Charger le modèle et le tokenizer
    # Make sure the model name is correctly specified in config.yaml if you are using a Hugging Face model
    # Based on the code, it seems you intend to use a HF model for fine-tuning.
    # Ensure your config.yaml has a 'hf_model' key under 'model'.
    # For example:
    # model:
    #   hf_model: "gpt2-medium" # or "gpt2", "gpt2-large", etc.
    #   ... other model parameters ...
    # If you are not using a Hugging Face model here, this part might need adjustment based on how load_model is supposed to work.
    # Assuming load_hf_model is the intended function for this script:
    # Ensure the config file is read correctly to provide the "hf_model" key
    try:
         model_name_from_config = config["model"]["hf_model"]
    except KeyError:
         print("Error: 'hf_model' not found under 'model' in config.yaml. Please add the Hugging Face model name.")
         return # Exit if config is missing the required key

    model, tokenizer, model_name = load_hf_model(config, device)
    print(f"Modèle chargé : {model_name}")

    # Préparer les datasets et dataloaders
    train_dataset = InstructionDataset(train_data, tokenizer)
    val_dataset = InstructionDataset(val_data, tokenizer)
    customized_collate_fn = partial(
        custom_collate_fn,
        device=device,
        allowed_max_length=config["model"]["context_length"],
        pad_token_id=tokenizer.pad_token_id
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["model"]["batch_size"],
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["model"]["batch_size"],
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False
    )

    # Calculer les pertes initiales
    train_loss = compute_loss(model, train_loader, device, num_batches=5)
    val_loss = compute_loss(model, val_loader, device, num_batches=5)
    print(f"Perte d'entraînement initiale : {train_loss:.4f}")
    print(f"Perte de validation initiale : {val_loss:.4f}")

    # Fine-tuning
    train_losses, val_losses = train_model(model, train_loader, val_loader, config, device)

    # Sauvegarder le modèle
    # Ensure the directory exists before saving
    import os
    save_dir = os.path.dirname(config["model"]["save_path"])
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(config["model"]["save_path"])
    tokenizer.save_pretrained(config["model"]["save_path"])
    print(f"Modèle sauvegardé sous {config['model']['save_path']}")

    # Sauvegarder le graphique des pertes
    # Ensure the directory exists before saving the plot
    plot_save_dir = os.path.dirname("/content/48_lois_pouvoir/loss-plot.pdf")
    os.makedirs(plot_save_dir, exist_ok=True)
    plot_losses(range(1, config["model"]["num_epochs"] + 1), train_losses, val_losses, "/content/48_lois_pouvoir/loss-plot.pdf")

    # Générer des réponses pour les données de test
    generate_responses(model, tokenizer, test_data, config, device)

if __name__ == "__main__":
    main()
