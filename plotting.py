from plotting_utils import simpan_figur
import matplotlib.pyplot as plt
from pathlib import Path
from plotting_utils import (
    BahanPlottinganModel,
    BahanPlottingan,
)
from local_type import (
    MatplotAxes,
)

def plot_loss(ax: MatplotAxes, plotting_loss: BahanPlottingan):
    ax.plot(plotting_loss.latihan, label="Loss Latih")
    ax.plot(plotting_loss.validasi, label="Loss Validasi")
    ax.set_title("Loss")
    ax.set_label("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()


def plot_akurasi(ax: MatplotAxes, plotting_akurasi: BahanPlottingan):
    ax.plot(plotting_akurasi.latihan, label="Loss Latih")
    ax.plot(plotting_akurasi.validasi, label="Loss Validasi")
    ax.set_title("Akurasi")
    ax.set_label("Epoch")
    ax.set_ylabel("Akurasi")
    ax.legend()

def tampilkan_plot_akur_loss(bahan_plottingan_model: BahanPlottinganModel, simpan: Path | None = None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    plot_akurasi(ax2, bahan_plottingan_model.plottingan_akurasi)
    plot_loss(ax1, bahan_plottingan_model.plottingan_loss)

    fig.canvas.manager.set_window_title("Plottingan Loss dan Akur")

    if simpan is not None:
        simpan_figur(fig, simpan.stem)

    plt.tight_layout()
    plt.show()