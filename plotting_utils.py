from local_const import EKSTENSI_IMGFIG
from local_type import MatplotFigure
from pathlib import Path
from utils import (
    pembuat_nama_model,
    pembuat_file
)

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
    figur.savefig(nama_file)