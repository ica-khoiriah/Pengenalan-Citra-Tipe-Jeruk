from local_const import KLASIFIKASI_NAMA_JERUK
from plotting_utils import simpan_figur
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from plotting_utils import (
    BahanPlottinganModel,
    BahanPlottinganData,
    BahanPlottingan,
)
from local_type import (
    LaporanKonfusiMatrix,
    KonfusiMatrix,
    Plottingan,
    MatplotAxes
)

def plot_loss(ax: MatplotAxes, plotting_loss: BahanPlottingan):
    ax.plot(plotting_loss.latihan, label="Loss Latih")
    ax.plot(plotting_loss.validasi, label="Loss Validasi")
    ax.set_title("Loss")
    ax.set_label("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

def plot_akurasi(ax: MatplotAxes, plotting_akurasi: BahanPlottingan):
    ax.plot(plotting_akurasi.latihan, label="Akurasi Latih")
    ax.plot(plotting_akurasi.validasi, label="Akurasi Validasi")
    ax.set_title("Akurasi")
    ax.set_label("Epoch")
    ax.set_ylabel("Akurasi")
    ax.legend()

def plot_data_loss(ax: MatplotAxes, plotting_loss: Plottingan):
    ax.plot(plotting_loss, label="Loss Data")
    # ax.plot(plotting_loss.validasi, label="Loss Validasi")
    ax.set_title("Loss")
    ax.set_label("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

def plot_data_akurasi(ax: MatplotAxes, plotting_akurasi: Plottingan):
    ax.plot(plotting_akurasi, label="Akurasi Data")
    # ax.plot(plotting_akurasi.validasi, label="Akurasi Validasi")
    ax.set_title("Akurasi")
    ax.set_label("Epoch")
    ax.set_ylabel("Akurasi")
    ax.legend()

def plot_konfusi_matrix(ax: MatplotAxes, konfusi_matrix: KonfusiMatrix):
    ax.imshow(konfusi_matrix, cmap="Wistia")
    panjang_klasifikasi = len(KLASIFIKASI_NAMA_JERUK)
    nama_jeruk = [nama.title() for nama in KLASIFIKASI_NAMA_JERUK.values()]

    ax.set_xticks(np.arange(panjang_klasifikasi))
    ax.set_yticks(np.arange(panjang_klasifikasi))
    ax.set_xticklabels(nama_jeruk)
    ax.set_yticklabels(nama_jeruk)

    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    ax.set_xlabel("Diprediksi")
    ax.set_ylabel("Sebenarnya")
    ax.set_title("Matriks Konfusi")

    for i in range(panjang_klasifikasi):
        for j in range(panjang_klasifikasi):
            ax.text(
                j, i,
                konfusi_matrix[i, j],
                ha="center",
                va="center",
                color="black"
            )
    
def tampilkan_plot_konfusi_matrix(
        kalkulasi_konfusi_matrix: KonfusiMatrix,
        simpan: Path | str | None = None
    ):
    fig, ax = plt.subplots(figsize=(10, 4))

    plot_konfusi_matrix(ax, kalkulasi_konfusi_matrix)

    fig.canvas.manager.set_window_title("Plottingan Matriks Konfusi")

    if simpan is not None:
        simpan_figur(fig, f"KonfusiMatrix_{simpan.stem if isinstance(simpan, Path) else simpan}")

    plt.tight_layout()

def tampilkan_plot_akur_loss_data(
        bahan_plottingan_model: BahanPlottinganData,
        simpan: Path | str | None = None
    ):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    plot_data_akurasi(ax2, bahan_plottingan_model.plottingan_akurasi)
    plot_data_loss(ax1, bahan_plottingan_model.plottingan_loss)

    fig.canvas.manager.set_window_title("Plottingan Loss dan Akur")

    if simpan is not None:
        simpan_figur(fig, f"AkurLoss_Data{simpan.stem if isinstance(simpan, Path) else simpan}")

    plt.tight_layout()

def tampilkan_plot_akur_loss(
        bahan_plottingan_model: BahanPlottinganModel,
        simpan: Path | str | None = None
    ):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    plot_akurasi(ax2, bahan_plottingan_model.plottingan_akurasi)
    plot_loss(ax1, bahan_plottingan_model.plottingan_loss)

    fig.canvas.manager.set_window_title("Plottingan Loss dan Akur")

    if simpan is not None:
        simpan_figur(fig, f"AkurLoss_{simpan.stem if isinstance(simpan, Path) else simpan}")

    plt.tight_layout()

def tampilkan_plot():
    plt.show()