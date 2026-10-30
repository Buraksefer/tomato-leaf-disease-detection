import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODELS = [
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

HISTORY_DIR = "outputs/history"
RESULTS_DIR = "outputs/results"
CM_DIR = "outputs/confusion_matrices"
SAVE_DIR = "outputs/comparison"

os.makedirs(SAVE_DIR, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def available_models():
    found = []
    for model in MODELS:
        history_path = os.path.join(HISTORY_DIR, f"{model}_history.json")
        result_path = os.path.join(RESULTS_DIR, f"{model}_report.json")
        cm_path = os.path.join(CM_DIR, f"{model}_cm.npy")

        if os.path.exists(history_path) and os.path.exists(result_path) and os.path.exists(cm_path):
            found.append(model)

    return found


def build_summary_table(models):
    rows = []

    for model in models:
        history_path = os.path.join(HISTORY_DIR, f"{model}_history.json")
        result_path = os.path.join(RESULTS_DIR, f"{model}_report.json")

        history = load_json(history_path)
        result = load_json(result_path)

        row = {
            "model": model,
            "best_val_acc": round(history["best_val_acc"], 2),
            "test_acc": round(result["accuracy"] * 100, 2),
            "macro_f1": round(result["f1_macro"] * 100, 2),
            "weighted_f1": round(result["f1_weighted"] * 100, 2),
            "final_train_acc": round(history["train_acc"][-1], 2),
            "final_val_acc": round(history["val_acc"][-1], 2),
            "epochs_ran": len(history["train_acc"]),
            "learning_rate_start": history["learning_rates"][0],
            "learning_rate_end": history["learning_rates"][-1],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["test_acc", "macro_f1", "best_val_acc"], ascending=False).reset_index(drop=True)
    return df


def save_summary(df):
    csv_path = os.path.join(SAVE_DIR, "summary_table.csv")
    txt_path = os.path.join(SAVE_DIR, "summary_table.txt")

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(df.to_string(index=False))

    return csv_path, txt_path


def plot_metric_comparison(df):
    x = np.arange(len(df))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, df["best_val_acc"], width=width, label="Best Val Acc")
    plt.bar(x, df["test_acc"], width=width, label="Test Acc")
    plt.bar(x + width, df["macro_f1"], width=width, label="Macro F1")

    plt.xticks(x, df["model"], rotation=20)
    plt.ylabel("Score (%)")
    plt.title("Model Performance Comparison")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "model_comparison.png"), dpi=300)
    plt.close()


def plot_overfitting_analysis(models):
    n = len(models)
    cols = 2
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    axes = np.array(axes).reshape(-1)

    for i, model in enumerate(models):
        history_path = os.path.join(HISTORY_DIR, f"{model}_history.json")
        history = load_json(history_path)

        epochs = list(range(1, len(history["train_loss"]) + 1))

        ax = axes[i]
        ax.plot(epochs, history["train_loss"], label="Train Loss")
        ax.plot(epochs, history["val_loss"], label="Val Loss")
        ax.set_title(f"{model} - Loss Curve")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "overfitting_loss_curves.png"), dpi=300)
    plt.close()


def plot_accuracy_curves(models):
    n = len(models)
    cols = 2
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    axes = np.array(axes).reshape(-1)

    for i, model in enumerate(models):
        history_path = os.path.join(HISTORY_DIR, f"{model}_history.json")
        history = load_json(history_path)

        epochs = list(range(1, len(history["train_acc"]) + 1))

        ax = axes[i]
        ax.plot(epochs, history["train_acc"], label="Train Acc")
        ax.plot(epochs, history["val_acc"], label="Val Acc")
        ax.set_title(f"{model} - Accuracy Curve")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.legend()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "accuracy_curves.png"), dpi=300)
    plt.close()


def plot_confusion_matrices(models):
    for model in models:
        cm_path = os.path.join(CM_DIR, f"{model}_cm.npy")
        result_path = os.path.join(RESULTS_DIR, f"{model}_report.json")

        cm = np.load(cm_path)
        result = load_json(result_path)
        class_names = result["class_names"]

        plt.figure(figsize=(10, 8))
        plt.imshow(cm)
        plt.title(f"{model} - Confusion Matrix")
        plt.colorbar()

        ticks = np.arange(len(class_names))
        plt.xticks(ticks, class_names, rotation=90)
        plt.yticks(ticks, class_names)

        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f"{model}_confusion_matrix.png"), dpi=300)
        plt.close()


def write_winner_report(df):
    winner = df.iloc[0]

    text = []
    text.append("MODEL KARŞILAŞTIRMA SONUCU\n")
    text.append(f"Sampiyon model: {winner['model']}\n")
    text.append(f"Best Validation Accuracy: {winner['best_val_acc']}%\n")
    text.append(f"Test Accuracy: {winner['test_acc']}%\n")
    text.append(f"Macro F1: {winner['macro_f1']}%\n")
    text.append(f"Weighted F1: {winner['weighted_f1']}%\n\n")

    text.append("OVERFITTING YORUMU (Genel Mantik):\n")
    text.append("- Train loss sürekli düşerken val loss sürekli yükseliyorsa overfitting vardır.\n")
    text.append("- Train ve val accuracy arasında çok büyük fark varsa overfitting riski yüksektir.\n")
    text.append("- En dengeli eğriye sahip model genelleme açısından daha güvenilirdir.\n\n")

    for _, row in df.iterrows():
        diff = round(row["final_train_acc"] - row["final_val_acc"], 2)
        text.append(
            f"{row['model']}: final train acc = {row['final_train_acc']}%, "
            f"final val acc = {row['final_val_acc']}%, fark = {diff}%\n"
        )

    save_path = os.path.join(SAVE_DIR, "winner_report.txt")
    with open(save_path, "w", encoding="utf-8") as f:
        f.writelines(text)

    return save_path


def main():
    models = available_models()

    if not models:
        print("Hic model sonucu bulunamadi. Once train/evaluate calistir.")
        return

    print("Bulunan modeller:", models)

    df = build_summary_table(models)
    csv_path, txt_path = save_summary(df)

    plot_metric_comparison(df)
    plot_overfitting_analysis(models)
    plot_accuracy_curves(models)
    plot_confusion_matrices(models)
    winner_path = write_winner_report(df)

    print("\n--- OZET TABLO ---")
    print(df.to_string(index=False))

    print("\nKaydedilen dosyalar:")
    print("CSV:", csv_path)
    print("TXT:", txt_path)
    print("Grafik: outputs/comparison/model_comparison.png")
    print("Grafik: outputs/comparison/overfitting_loss_curves.png")
    print("Grafik: outputs/comparison/accuracy_curves.png")
    print("Confusion matrix'ler: outputs/comparison/*_confusion_matrix.png")
    print("Winner raporu:", winner_path)


if __name__ == "__main__":
    main()