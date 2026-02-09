from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Tuple
from pathlib import Path
import torch.nn as tnn
import torch

@dataclass(frozen=True)
class PerangkatTrainingModel:
    CUDA: str = "cuda"
    GPU: str = "cuda"
    CPU: str = "cpu"

type NamaPerangkat = str
type DataLoaderLatih = DataLoader
type DataLoaderValidasi = DataLoader

def perangkat_pilihan() -> NamaPerangkat:
    return (
        PerangkatTrainingModel.CUDA
        if torch.cuda.is_available()
        else PerangkatTrainingModel.CPU
    )

def muat_dataset() -> Tuple[DataLoaderLatih, DataLoaderValidasi]:
    # data-nya di augmentasikan
    transformasi_data_latih = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(32, scale=(.8, 1.0)),
        transforms.ColorJitter(brightness=.2, contrast=.2),
        transforms.ToTensor(),
    ])

    transformasi_data = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    kumpulan_data_latih = datasets.ImageFolder(
        root="./dataset/train",
        transform=transformasi_data_latih
    )

    kumpulan_data_validasi = datasets.ImageFolder(
        root="./dataset/val",
        transform=transformasi_data
    )

    pemuat_data_latih = DataLoader(
        kumpulan_data_latih,
        batch_size=16,
        shuffle=True
    )

    pemuat_data_validasi = DataLoader(
        kumpulan_data_validasi,
        batch_size=16,
        shuffle=False
    )

    return pemuat_data_latih, pemuat_data_validasi

def simpan_model(model: tnn.Module, nama: str = "model", ekstensi: str = "pth"):
    torch.save(model.state_dict(), Path("./model/") / Path(f"{nama}.{ekstensi}"))

def muat_model(model: tnn.Module, path_citra: Path):
    model.load_state_dict(torch.load(path_citra, map_location=perangkat_pilihan()))
    return model