from model_transforms import ModelTransform
from torch.utils.data import DataLoader
from torchvision import datasets
from pathlib import Path
from typing import Tuple
import torch
from dataclasses import dataclass
from plotting_utils import (
    BahanPlottinganModel,
    BahanPlottinganData,
)
from utils import (
    pembuat_nama_model,
    pembuat_file,
)
from torch.utils.data import DataLoader
from local_const import (
    PerangkatTrainingModel,
    FOLDER_TRAINING,
    FOLDER_VALIDASI,
    FOLDER_MODEL,
    EKSTENSI_MODEL,
)
from local_type import (
    DataLoaderValidasi,
    DataLoaderLatih,
    NamaPerangkat,
    Model,
)

@dataclass
class HasilLatih:
    bahan_plottingan_model: BahanPlottinganModel
    nama_model: str | None = None

@dataclass
class HasilPrediksi:
    bahan_plottingan_data: BahanPlottinganData
    nama_prediksi: str | None = None

def perangkat_pilihan() -> NamaPerangkat:
    return (
        PerangkatTrainingModel.CUDA
        if torch.cuda.is_available()
        else PerangkatTrainingModel.CPU
    )

def muat_dataset() -> Tuple[DataLoaderLatih, DataLoaderValidasi]:
    kumpulan_data_latih = datasets.ImageFolder(
        root=FOLDER_TRAINING,
        transform=ModelTransform.transform_latih
    )

    kumpulan_data_validasi = datasets.ImageFolder(
        root=FOLDER_VALIDASI,
        transform=ModelTransform.transform_validasi
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

def muat_data(folder_path: Path) -> DataLoader:
    kumpulan_data = datasets.ImageFolder(
        root=folder_path,
        transform=ModelTransform.transform_validasi
    )

    pemuat_data = DataLoader(
        kumpulan_data,
        batch_size=2,
        shuffle=False,
    )

    return pemuat_data

def simpan_model(
        model: Model,
        nama_model: str | None = None,
        ekstensi: str = EKSTENSI_MODEL
    ) -> Path:
    nama_model = (
        pembuat_nama_model()
        if nama_model is None
        else nama_model
    )
    nama_file = pembuat_file(nama_model, ekstensi)
    torch.save(
        model.state_dict(),
        FOLDER_MODEL / nama_file
    )
    return nama_file

def muat_model(model: Model, path_citra: Path):
    model.load_state_dict(
        torch.load(
            path_citra,
            map_location=perangkat_pilihan()
        )
    )
    return model