from pathlib import Path
from utils import ambil_file
from local_const import KLASIFIKASI_NAMA_JERUK
from model_transforms import ModelTransform
from model_utils import perangkat_pilihan
from torch.utils.data import DataLoader
from PIL import Image
import torch
from local_type import (
    Model,
)

def prediksi_banyak(model: Model, pemuat_latih: DataLoader):
    semua_prediksi = []
    semua_label = []

    perang_latih = torch.device(perangkat_pilihan())

    with torch.no_grad():
        # for gambar, label 
        pass