import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

def compute_loss(model, loader, device, num_batches=None):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, labels=inputs)
            loss = outputs.loss
            total_loss += loss.item()
            count += 1
            if num_batches and count >= num_batches:
                break
    return total_loss / count

def train_model(model, train_loader, val_loader, config, device):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["model"]["learning_rate"],
        weight_decay=config["model"]["weight_decay"]
    )
    num_epochs = config["model"]["num_epochs"]
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for inputs, targets in tqdm(train_loader, desc=f"Époque {epoch+1}/{num_epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, labels=inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

        val_loss = compute_loss(model, val_loader, device)
        val_losses.append(val_loss)
        print(f"Époque {epoch+1}/{num_epochs} - Perte entraînement : {train_losses[-1]:.4f}, Perte validation : {val_loss:.4f}")

    return train_losses, val_losses

def plot_losses(epochs_seen, train_losses, val_losses, save_path):
    plt.figure(figsize=(12, 6))
    plt.plot(epochs_seen, train_losses, label="Perte d'entraînement")
    plt.plot(epochs_seen, val_losses, linestyle="-.", label="Perte de validation")
    plt.xlabel("Époques")
    plt.ylabel("Perte")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Graphique sauvegardé sous {save_path}")
