from datetime import datetime
from pathlib import Path

def pembuat_file(nama_file: str, ekstensi: str) -> Path:
    return Path(f"{nama_file}.{ekstensi}")

def pembuat_nama_model() -> str:
    waktu = datetime.now()
    nama_model = f"model-{waktu.day}_{waktu.month}_{waktu.strftime('%y')}-v{waktu.strftime('%H%M%S')}"
    return nama_model

def ambil_file_terbaru(path_folder: Path) -> Path | None:
    files = [
        f for f in path_folder.iterdir() if f.is_file()
    ]

    if not files:
        return None

    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    return latest_file