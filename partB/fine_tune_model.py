# finetune_model.py

import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import os
from torchvision.models import GoogLeNet_Weights


def get_dataloaders(img_size, batch_size):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    data_dir = "../inaturalist_12K"
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader

def apply_freezing_strategy(model, strategy):
    if strategy == "only_fc":
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

    elif strategy == "partial":
        freeze_until = 10  # Freeze up to a few blocks
        children = list(model.children())
        for i, child in enumerate(children):
            for param in child.parameters():
                param.requires_grad = i >= freeze_until

    elif strategy == "all":
        for param in model.parameters():
            param.requires_grad = True

def train():
    # Initializing wandb
    wandb.init()
    config = wandb.config

    run_name = f"finetune_{config.freeze_strategy}_bs{config.batch_size}_lr{config.learning_rate}"
    wandb.run.name = run_name
    #wandb.run.save()

    # Loading data
    train_loader, val_loader = get_dataloaders(config.img_size, config.batch_size)

    # Loading and modify pre-trained GoogLeNet
    weights = GoogLeNet_Weights.DEFAULT
    model = models.googlenet(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 10)

    apply_freezing_strategy(model, config.freeze_strategy)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)


    # Training loop
    for epoch in range(config.epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"Epoch [{epoch+1}/{config.epochs}] | Train Acc: {train_acc:.2%}, Val Acc: {val_acc:.2%}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        # Saving model if it's the best across all sweeps
        global_best_path = "best_accuracy_B.txt"
        current_best = 0.0

        if os.path.exists(global_best_path):
            with open(global_best_path, "r") as f:
                try:
                    current_best = float(f.read().strip())
                except:
                    current_best = 0.0

        if val_acc > current_best:
            torch.save(model.state_dict(), "best_model_B.pth")
            with open(global_best_path, "w") as f:
                f.write(str(val_acc))
            print(f"New global best model saved with val_acc: {val_acc:.4f}")

    print("Training complete.")
    wandb.finish()


# For sweep usage
if __name__ == "__main__":
    from sweep_config_2 import sweep_config
    sweep_id = wandb.sweep(sweep_config, project="DA6401_assign_2")
    wandb.agent(sweep_id, function=train, count=1)
    wandb.finish()