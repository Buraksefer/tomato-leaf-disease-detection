import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from models.resnet_transfer import ResNetTransfer


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_hook = self.target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach().clone()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach().clone()

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[:, class_idx]
        score.backward()

        gradients = self.gradients[0]      # [C, H, W]
        activations = self.activations[0]  # [C, H, W]

        weights = gradients.mean(dim=(1, 2))

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32).to(activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam.cpu().numpy(), class_idx

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    return image, tensor


def overlay_cam_on_image(pil_image, cam):
    image = np.array(pil_image.resize((224, 224))).astype(np.float32) / 255.0

    # CAM'i 224x224 boyutuna büyüt
    cam_resized = cv2.resize(cam, (224, 224))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    overlay = 0.6 * image + 0.4 * heatmap
    overlay = np.clip(overlay, 0, 1)

    return image, heatmap, overlay


def main():
    class_names = [
        'Tomato___Bacterial_spot',
        'Tomato___Early_blight',
        'Tomato___Late_blight',
        'Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'
    ]

    image_path = "Data/tomato/test/Tomato___Early_blight/c10623b4-33fe-458c-9d8e-d9a74fb74a6a___RS_Erly.B 7450.JPG"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNetTransfer(num_classes=len(class_names), fine_tune=True).to(device)
    model.load_state_dict(
        torch.load("outputs/models/best_densenet_finetune.pth", map_location=device)
    )
    model.eval()

    # DÜZELTİLEN SATIR
    target_layer = model.model.features.denseblock4

    gradcam = GradCAM(model, target_layer)

    pil_image, input_tensor = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)

    cam, pred_idx = gradcam.generate(input_tensor)
    predicted_class = class_names[pred_idx]

    print("Tahmin edilen sinif:", predicted_class)

    original, heatmap, overlay = overlay_cam_on_image(pil_image, cam)

    os.makedirs("outputs/gradcam", exist_ok=True)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title("Orijinal Görsel")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.title("Grad-CAM Heatmap")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title(f"Overlay\nTahmin: {predicted_class}")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("outputs/gradcam/gradcam_result.png", dpi=300)
    plt.show()

    gradcam.remove_hooks()


if __name__ == "__main__":
    main()