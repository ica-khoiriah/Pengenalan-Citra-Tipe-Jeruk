from model_utils import (
    HasilPrediksi,
    perangkat_pilihan
)
from plotting_utils import (
    BahanPlottinganData,
    BahanPlottinganMatriksKonfusi
)
from torch.utils.data import DataLoader
import torch.nn as tnn
from tqdm import tqdm
import torch
from local_type import (
    Model,
)
from local_const import (
    TDQM_BAR_FORMAT,
    TDQM_BAR_STYLE
)

def prediksi_banyak(
        model: Model,
        pemuat_data: DataLoader,
        nama_prediksi: str
    ) -> HasilPrediksi:
    semua_prediksi = []
    semua_label = []
    daftar_loss = []
    daftar_akur = []
    kumpulan_loss = .0
    total = 0
    benar = 0

    perangkat_prediksi = torch.device(perangkat_pilihan())
    kriterion = tnn.CrossEntropyLoss()

    with torch.no_grad():
        loop = tqdm(
            pemuat_data,
            desc=f"[{nama_prediksi}]",
            total=len(pemuat_data),
            bar_format=TDQM_BAR_FORMAT,
            ascii=TDQM_BAR_STYLE
        )

        for gambar, label in loop:
            gambar = gambar.to(perangkat_prediksi)
            label = label.to(perangkat_prediksi)

            output = model(gambar)
            loss = kriterion(output, label)

            kumpulan_loss += loss.item()

            _, prediksi = torch.max(output, 1)
            total += label.size(0)
            benar += (prediksi == label).sum().item()

            akurasi = 100.0 * benar / total
            rata2_loss = kumpulan_loss / total

            semua_prediksi.extend(prediksi.cpu().numpy())
            semua_label.extend(label.cpu().numpy())

            loop.set_postfix({
                "loss": f"{rata2_loss:.4f}",
                "akur": f"{akurasi:.2f}%"
            })

            akurasi_data = 100 * benar / total
            loss_data = kumpulan_loss / len(pemuat_data)

            daftar_loss.append(akurasi_data)
            daftar_akur.append(loss_data)

    hasil_predik = HasilPrediksi(
        bahan_plottingan_data=BahanPlottinganData(
            plottingan_akurasi=daftar_akur,
            plottingan_loss=daftar_loss,
            plottingan_prediksi=BahanPlottinganMatriksKonfusi(
                prediksi=semua_prediksi,
                label=semua_label
            )
        ),
        nama_prediksi=nama_prediksi
    )

    return hasil_predik
