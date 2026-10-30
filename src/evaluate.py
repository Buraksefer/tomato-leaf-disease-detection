import os
import json
import argparse
import numpy as np
import torch
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)

from dataset import get_dataloaders
from model_factory import get_model


ALL_MODELS = [
    "basic_cnn",
    "improved_cnn",
    "deeper_cnn",
    "densenet",
    "resnet",
    "mobilenet",
    "vit",
    "swin",
    "deit"
]


def save_json_report(save_path, data):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def evaluate_model(model_name, device):
    train_dir = "Data/tomato/train"
    val_dir = "Data/tomato/val"
    test_dir = "Data/tomato/test"

    batch_size = 16 if model_name in ["vit", "swin", "deit"] else 32

    _, _, test_loader, class_names = get_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        batch_size=batch_size
    )

    model = get_model(model_name, len(class_names)).to(device)

    model_path = f"outputs/models/best_{model_name}.pth"
    if not os.path.exists(model_path):
        print(f"Model bulunamadı, atlandı: {model_path}")
        return None

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0
    )

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0
    )

    report_text = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        zero_division=0
    )

    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(all_labels, all_preds)

    os.makedirs("outputs/results", exist_ok=True)
    os.makedirs("outputs/confusion_matrices", exist_ok=True)

    txt_path = f"outputs/results/{model_name}_report.txt"
    json_path = f"outputs/results/{model_name}_report.json"
    cm_path = f"outputs/confusion_matrices/{model_name}_cm.npy"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    save_json_report(json_path, {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "class_names": class_names,
        "classification_report": report_dict
    })

    np.save(cm_path, cm)

    print("\n" + "=" * 60)
    print(f"Model: {model_name}")
    print("=" * 60)
    print(report_text)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro Precision: {precision_macro * 100:.2f}%")
    print(f"Macro Recall: {recall_macro * 100:.2f}%")
    print(f"Macro F1: {f1_macro * 100:.2f}%")

    return {
        "model": model_name,
        "accuracy": round(accuracy * 100, 2),
        "precision_macro": round(precision_macro * 100, 2),
        "recall_macro": round(recall_macro * 100, 2),
        "f1_macro": round(f1_macro * 100, 2),
        "precision_weighted": round(precision_weighted * 100, 2),
        "recall_weighted": round(recall_weighted * 100, 2),
        "f1_weighted": round(f1_weighted * 100, 2)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=ALL_MODELS + ["all"]
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Kullanılan cihaz:", device)

    if args.model == "all":
        models_to_run = ALL_MODELS
    else:
        models_to_run = [args.model]

    results = []

    for model_name in models_to_run:
        result = evaluate_model(model_name, device)
        if result is not None:
            results.append(result)

    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(by="accuracy", ascending=False)

        os.makedirs("outputs/results", exist_ok=True)
        summary_path = "outputs/results/summary_metrics.csv"
        df.to_csv(summary_path, index=False, encoding="utf-8-sig")

        print("\nGENEL MODEL KARŞILAŞTIRMA TABLOSU")
        print(df.to_string(index=False))
        print(f"\nÖzet tablo kaydedildi: {summary_path}")


if __name__ == "__main__":
    main()