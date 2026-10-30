import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from dataset import get_dataloaders
from model_factory import get_model


def get_training_config(model_name: str) -> dict:

    configs = {
        "basic_cnn": {
            "batch_size": 32,
            "num_epochs": 10,
            "learning_rate": 0.001,
            "patience": 5,
            "use_scheduler": False
        },
        "improved_cnn": {
            "batch_size": 32,
            "num_epochs": 20,
            "learning_rate": 0.001,
            "patience": 5,
            "use_scheduler": True
        },
        "deeper_cnn": {
            "batch_size": 32,
            "num_epochs": 30,
            "learning_rate": 0.001,
            "patience": 5,
            "use_scheduler": True
        },
        "densenet": {
            "batch_size": 32,
            "num_epochs": 15,
            "learning_rate": 0.0001,
            "patience": 5,
            "use_scheduler": True
        },
        "resnet": {
            "batch_size": 32,
            "num_epochs": 15,
            "learning_rate": 0.0001,
            "patience": 5,
            "use_scheduler": True
        },
        "mobilenet": {
            "batch_size": 32,
            "num_epochs": 15,
            "learning_rate": 0.0001,
            "patience": 5,
            "use_scheduler": True
        },
        "vit": {
            "batch_size": 16,
            "num_epochs": 15,
            "learning_rate": 0.0001,
            "patience": 5,
            "use_scheduler": True
        },
        "swin": {
            "batch_size": 16,
            "num_epochs": 15,
            "learning_rate": 0.0001,
            "patience": 5,
            "use_scheduler": True
        },
        "deit": {
            "batch_size": 16,
            "num_epochs": 15,
            "learning_rate": 0.0001,
            "patience": 5,
            "use_scheduler": True
        }
    }

    if model_name not in configs:
        raise ValueError(f"Desteklenmeyen model adı: {model_name}")

    return configs[model_name]


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Egitim", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def save_history(history: dict, save_path: str):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["basic_cnn", "improved_cnn", "deeper_cnn", "densenet", "resnet", "mobilenet", "vit", "swin", "deit"]
    )
    args = parser.parse_args()

    model_name = args.model
    config = get_training_config(model_name)

    train_dir = "Data/tomato/train"
    val_dir = "Data/tomato/val"
    test_dir = "Data/tomato/test"

    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/history", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Kullanilan cihaz:", device)
    print("Secilen model:", model_name)
    print("Egitim ayarlari:", config)

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        batch_size=config["batch_size"]
    )

    print("Siniflar:", class_names)

    model = get_model(model_name, len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"]
    )

    scheduler = None
    if config["use_scheduler"]:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2
        )

    best_val_acc = 0.0
    early_stop_counter = 0

    history = {
        "model_name": model_name,
        "config": config,
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rates": [],
        "best_val_acc": 0.0,
        "class_names": class_names
    }

    model_save_path = f"outputs/models/best_{model_name}.pth"
    history_save_path = f"outputs/history/{model_name}_history.json"

    for epoch in range(config["num_epochs"]):
        print(f"\nEpoch [{epoch + 1}/{config['num_epochs']}]")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["learning_rates"].append(current_lr)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")
        print(f"Current LR: {current_lr:.6f}")

        if scheduler is not None:
            scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            history["best_val_acc"] = best_val_acc
            early_stop_counter = 0

            torch.save(model.state_dict(), model_save_path)
            print("En iyi model kaydedildi.")
        else:
            early_stop_counter += 1
            print(f"Early stopping sayaci: {early_stop_counter}/{config['patience']}")

        save_history(history, history_save_path)

        if early_stop_counter >= config["patience"]:
            print("\nEarly stopping tetiklendi. Egitim durduruldu.")
            break

    print(f"\nEn iyi validation accuracy: {best_val_acc:.2f}%")
    print(f"Model kayit yolu: {model_save_path}")
    print(f"History kayit yolu: {history_save_path}")


if __name__ == "__main__":
    main()