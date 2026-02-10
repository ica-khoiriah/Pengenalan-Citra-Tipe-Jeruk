from torchvision import transforms as transf
from local_const import TRANSFORM_SIZE
from local_type import TipeCompose
from dataclasses import dataclass

@dataclass
class ModelTransform:
    transform_latih: TipeCompose = transf.Compose([
        transf.RandomResizedCrop(TRANSFORM_SIZE, scale=(.6, 1.0)),
        transf.RandomHorizontalFlip(),
        transf.RandomRotation(15),
        transf.ColorJitter(
            brightness=.3,
            contrast=.2,
            saturation=.3,
            hue=.05
        ),
        transf.ToTensor(),
    ])
    transform_validasi: TipeCompose = transf.Compose([
        transf.Resize(int(TRANSFORM_SIZE * 1.14)),
        transf.CenterCrop(TRANSFORM_SIZE),
        transf.ToTensor(),
    ])

