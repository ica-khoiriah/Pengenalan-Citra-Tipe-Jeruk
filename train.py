from model_utils import simpan_model, perangkat_pilihan
from torch.utils.data import DataLoader
from local_type import DaftarPlottingan
from dataclasses import dataclass
from typing import Tuple, Final
import torch.optim as toptim
import torch.nn as tnn
import torch

JUMLAH_EPOCH: Final[int] = 10 # 10 epoch biar cukup

@dataclass
class SimpanModul:
    nama: str
    ekstensi: str = "pth"

def latih_model(
        model: tnn.Module,
        pemuat_latih: DataLoader,
        pemuat_validasi: DataLoader,
        simpan: SimpanModul | None = None
    ) -> Tuple[
        DaftarPlottingan,
        DaftarPlottingan,
        DaftarPlottingan,
        DaftarPlottingan
    ]:
    perangkat_latih = torch.device(perangkat_pilihan())

    model = model.to(perangkat_latih)
    kriterion = tnn.CrossEntropyLoss()
    optimiser = toptim.Adam(model.parameters(), lr=.001)

    daftar_loss_latih = []
    daftar_loss_validasi = []
    daftar_akurasi_latih = []
    daftar_akurasi_validasi = []

    for epoch in range(JUMLAH_EPOCH):
        model.train()
        kumpulan_latihan_loss = .0
        kumpulan_validasi_loss = .0
        benar = 0
        total = 0

        # latihan
        for gambar, label in pemuat_latih:
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
        
        akurasi_latih = 100 * benar / total
        loss_latih = kumpulan_latihan_loss / len(pemuat_latih)

        daftar_akurasi_latih.append(akurasi_latih)
        daftar_loss_latih.append(loss_latih)

        model.eval()
        benar_validasi = 0
        total_validasi = 0

        # validasi
        with torch.no_grad():
            for gambar, label in pemuat_validasi:
                gambar = gambar.to(perangkat_latih)
                label = label.to(perangkat_latih)

                output = model(gambar)
                loss = kriterion(output, label)

                kumpulan_validasi_loss += loss.item()

                _, prediksi = torch.max(output, 1)
                total_validasi += label.size(0)
                benar_validasi += (prediksi == label).sum().item()

        akurasi_validasi = 100 * benar_validasi / total_validasi
        loss_latih = kumpulan_validasi_loss / len(pemuat_validasi)

        daftar_akurasi_validasi.append(akurasi_validasi)
        daftar_loss_validasi.append(loss_latih)

        print(f"Epoch [{epoch+1}/{JUMLAH_EPOCH}] "
              f"Loss: {loss_latih:.2f} ",
              f"Akurasi: {akurasi_latih:.2f}% "
              f"Akurasi Validasi: {akurasi_validasi:.2f}%"
        )

    if simpan is not None:
        simpan_model(
            model,
            simpan.nama,
            simpan.ekstensi
        )

    return (
        tuple(daftar_akurasi_latih),
        tuple(daftar_akurasi_validasi),
        tuple(daftar_loss_latih),
        tuple(daftar_loss_validasi)
    )