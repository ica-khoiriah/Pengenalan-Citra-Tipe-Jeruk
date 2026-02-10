from dataclasses import dataclass
from pathlib import Path
from typing import (
    Final,
    Dict
)

TRANSFORM_SIZE: Final[int] = 64
EKSTENSI_MODEL: Final[str] = "pth"
EKSTENSI_IMGFIG: Final[str] = "png"
FOLDER_MODEL: Final[Path] = Path("./model/")
FOLDER_DATASET: Final[Path] = Path("./dataset/")
FOLDER_TRAINING: Final[Path] = FOLDER_DATASET / "train"
FOLDER_VALIDASI: Final[Path] = FOLDER_DATASET / "val"
KLASIFIKASI_NAMA_JERUK: Final[Dict[int, str]] = {
    0: "gerga",
    1: "kalamansi"
}

@dataclass(frozen=True)
class PerangkatTrainingModel:
    CUDA: str = "cuda"
    GPU: str = "cuda"
    CPU: str = "cpu"