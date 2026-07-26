from architectures.CNNs.LeNet_5 import Lenet
from architectures.CNNs.VGG_ResNet_Hybrid import PinteeCNN
from architectures.CNNs.ConvNeXt_v1_Adapted import ConvNeXt_V1


MODEL_REGISTRY = {
    "pintee_cnn" : PinteeCNN,
    "lenet"      : Lenet,
    "convnext_v1": ConvNeXt_V1
}