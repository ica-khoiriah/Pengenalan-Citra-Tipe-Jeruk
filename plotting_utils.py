from local_const import (
    EKSTENSI_IMGFIG,
    FOLDER_FIGUR
)
from local_type import MatplotFigure
from dataclasses import dataclass
from local_type import Plottingan
from pathlib import Path
from utils import (
    pembuat_nama_model,
    pembuat_file
)

@dataclass
class BahanPlottingan:
    latihan: Plottingan
    validasi: Plottingan

@dataclass
class BahanPlottinganMatriksKonfusi:
    prediksi: Plottingan
    label: Plottingan

@dataclass
class BahanPlottinganModel:
    plottingan_akurasi: BahanPlottingan
    plottingan_loss: BahanPlottingan
    plottingan_prediksi: BahanPlottinganMatriksKonfusi

def simpan_figur(
        figur: MatplotFigure,
        nama_figur: str | None = None,
        ekstensi: str = EKSTENSI_IMGFIG
    ) -> Path:
    nama_figur = (
        pembuat_nama_model()
        if nama_figur is None
        else nama_figur
    )
    nama_file = pembuat_file(nama_figur, ekstensi)
    figur.savefig(FOLDER_FIGUR / nama_file)