import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def preprocess_pil_image(pil_image: Image.Image):
    image = pil_image.convert("RGB").resize((224, 224))

    rgb_image = np.array(image).astype(np.float32) / 255.0

    transform = get_transform()
    input_tensor = transform(image).unsqueeze(0)

    return image, rgb_image, input_tensor


def generate_gradcam_for_pil(
    model,
    pil_image,
    target_layer,
    device="cpu",
    method="gradcam++"
):
    model = model.to(device)
    model.eval()

    pil_image, rgb_image, input_tensor = preprocess_pil_image(pil_image)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        pred_idx = outputs.argmax(dim=1).item()

    if method == "gradcam++":
        cam_algorithm = GradCAMPlusPlus
    elif method == "eigencam":
        cam_algorithm = EigenCAM
    else:
        cam_algorithm = GradCAM

    cam = cam_algorithm(
        model=model,
        target_layers=[target_layer]
    )

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=None
    )

    grayscale_cam = grayscale_cam[0, :]

    overlay = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    heatmap = cv2.applyColorMap(
        np.uint8(255 * grayscale_cam),
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return {
        "original": rgb_image,
        "heatmap": heatmap,
        "overlay": overlay,
        "pred_idx": pred_idx,
        "pred_class": CLASS_NAMES[pred_idx]
    }