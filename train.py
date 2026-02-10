from torch.utils.data import DataLoader
import torch.optim as toptim
from pathlib import Path
import torch.nn as tnn
from tqdm import tqdm
import torch
from typing import (
    Tuple,
    Final
)
from model_utils import (
    simpan_model,
    perangkat_pilihan
)
from local_type import (
    Plottingan,
    Model,
)

JUMLAH_EPOCH: Final[int] = 10                                   # 10: default - biar cukup.

def latih_model(
        model: Model,
        pemuat_latih: DataLoader,
        pemuat_validasi: DataLoader,
        simpan: bool = True,
        jumlah_epoch: int | None = None
    ) -> Tuple[
        Plottingan,
        Plottingan,
        Plottingan,
        Plottingan,
    ] | Tuple[
        Plottingan,
        Plottingan,
        Plottingan,
        Plottingan,
        Path
    ]:
    perangkat_latih = torch.device(perangkat_pilihan())

    model = model.to(perangkat_latih)
    kriterion = tnn.CrossEntropyLoss()
    optimiser = toptim.Adam(model.parameters(), lr=.001)

    daftar_loss_latih = []
    daftar_loss_validasi = []
    daftar_akurasi_latih = []
    daftar_akurasi_validasi = []

    if jumlah_epoch is None:
        jumlah_epoch = JUMLAH_EPOCH

    for epoch in range(jumlah_epoch):
        model.train()
        kumpulan_latihan_loss = .0
        kumpulan_validasi_loss = .0
        benar = 0
        total = 0

        loop = tqdm(pemuat_latih, desc=f"Epoch {epoch + 1}/{jumlah_epoch} [Latih]")

        # latihan
        for gambar, label in loop:
            gambar = gambar.to(perangkat_latih)
            label = label.to(perangkat_latih)

            output = model(gambar)
            loss = kriterion(output, label)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            kumpulan_latihan_loss += loss.item()

            _, prediksi = torch.max(output, 1)
            total += label.size(0)
            benar += (prediksi == label).sum().item()

            akurasi = 100.0 * benar / total
            rata2_loss = kumpulan_latihan_loss / total

            loop.set_postfix({
                "loss": f"{rata2_loss:.4f}",
                "akur": f"{akurasi:.2f}%" 
            })
        
        akurasi_latih = 100 * benar / total
        loss_latih = kumpulan_latihan_loss / len(pemuat_latih)

        daftar_akurasi_latih.append(akurasi_latih)
        daftar_loss_latih.append(loss_latih)

        model.eval()
        benar_validasi = 0
        total_validasi = 0

        # validasi
        with torch.no_grad():
            loop = tqdm(pemuat_latih, desc=f"Epoch {epoch + 1}/{jumlah_epoch} [Valid]")

            for gambar, label in loop:
                gambar = gambar.to(perangkat_latih)
                label = label.to(perangkat_latih)

                output = model(gambar)
                loss = kriterion(output, label)

                kumpulan_validasi_loss += loss.item()

                _, prediksi = torch.max(output, 1)
                total_validasi += label.size(0)
                benar_validasi += (prediksi == label).sum().item()

                akurasi = 100.0 * benar / total
                rata2_loss = kumpulan_validasi_loss / total

                loop.set_postfix({
                    "loss": f"{rata2_loss:.4f}",
                    "akur": f"{akurasi:.2f}%" 
                })

        akurasi_validasi = 100 * benar_validasi / total_validasi
        loss_latih = kumpulan_validasi_loss / len(pemuat_validasi)

        daftar_akurasi_validasi.append(akurasi_validasi)
        daftar_loss_validasi.append(loss_latih)

        # print(f"Epoch [{epoch+1}/{jumlah_epoch}] "
        #       f"Loss: {loss_latih:.2f} ",
        #       f"Akurasi: {akurasi_latih:.2f}% "
        #       f"Akurasi Validasi: {akurasi_validasi:.2f}%"
        # )

    if simpan:
        nama_model = simpan_model(model)
        return (
            tuple(daftar_akurasi_latih),
            tuple(daftar_akurasi_validasi),
            tuple(daftar_loss_latih),
            tuple(daftar_loss_validasi),
            nama_model
        )
    else:
        return (
            tuple(daftar_akurasi_latih),
            tuple(daftar_akurasi_validasi),
            tuple(daftar_loss_latih),
            tuple(daftar_loss_validasi)
        )