from models.basic_cnn import BasicCNN
from models.improved_cnn import ImprovedCNN
from models.deeper_cnn import DeeperCNN
from models.densenet_transfer import DenseNetTransfer
from models.resnet_transfer import ResNetTransfer
from models.mobilenet_transfer import MobileNetTransfer
from models.vit_transfer import ViTTransfer
from models.swin_transfer import SwinTransfer
from models.deit_transfer import DeiTTransfer


def get_model(model_name, num_classes):
    model_name = model_name.lower()

    if model_name == "basic_cnn":
        return BasicCNN(num_classes=num_classes)

    elif model_name == "improved_cnn":
        return ImprovedCNN(num_classes=num_classes)

    elif model_name == "deeper_cnn":
        return DeeperCNN(num_classes=num_classes)

    elif model_name == "densenet":
        return DenseNetTransfer(num_classes=num_classes, fine_tune=True)

    elif model_name == "resnet":
        return ResNetTransfer(num_classes=num_classes, fine_tune=True)

    elif model_name == "mobilenet":
        return MobileNetTransfer(num_classes=num_classes, fine_tune=True)
    
    elif model_name == "vit":
        return ViTTransfer(num_classes=num_classes, fine_tune=True)

    elif model_name == "swin":
        return SwinTransfer(num_classes=num_classes, fine_tune=True)

    elif model_name == "deit":
        return DeiTTransfer(num_classes=num_classes, fine_tune=True)

    else:
        raise ValueError(f"Desteklenmeyen model adı: {model_name}")
    
    