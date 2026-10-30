import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model_factory import get_model


EXTERNAL_TEST_DIR = "Data/external_plantvillage"
MODEL_PATH = "outputs/models/best_densenet.pth"

BATCH_SIZE = 32


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Kullanılan cihaz:", device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = datasets.ImageFolder(
        EXTERNAL_TEST_DIR,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print("\nExternal sınıflar:")
    print(dataset.classes)

    model = get_model(
        "densenet",
        num_classes=len(dataset.classes)
    ).to(device)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            outputs = model(images)

            preds = outputs.argmax(dim=1)

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

    acc = accuracy_score(
        all_labels,
        all_preds
    )

    cm = confusion_matrix(
        all_labels,
        all_preds
    )

    print("\nDenseNet External Test Accuracy:")
    print(f"{acc * 100:.2f}%")

    print("\nClassification Report:")

    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=dataset.classes,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    print(cm)

    plt.figure(figsize=(14, 12))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=dataset.classes,
        yticklabels=dataset.classes
    )

    plt.title(
        "DenseNet External Test Confusion Matrix",
        fontsize=16
    )

    plt.xlabel(
        "Predicted Label",
        fontsize=12
    )

    plt.ylabel(
        "True Label",
        fontsize=12
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.yticks(
        rotation=0
    )

    plt.tight_layout()

    output_file = (
        "outputs/densenet_external_confusion_matrix.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        f"\nKarmaşıklık matrisi kaydedildi: {output_file}"
    )

    plt.show()


if __name__ == "__main__":
    main()