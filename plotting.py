from local_type import DaftarPlottingan
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class BahanPlottinganModel:
    akurasi_latihan: DaftarPlottingan
    akurasi_validasi: DaftarPlottingan
    loss_latihan: DaftarPlottingan
    loss_validasi: DaftarPlottingan

__bahan_plottingan_model: BahanPlottinganModel | None = None

def siapkan_plottingan(bahan_plottingan_model: BahanPlottinganModel):
    global __bahan_plottingan_model
    __bahan_plottingan_model = bahan_plottingan_model

def __siapkan_figure():
    plt.figure()
    plt.tight_layout()

def plotting_loss(figure_sama: bool = True):
    if not figure_sama:
        plt.figure()
    plt.plot(__bahan_plottingan_model.loss_latihan, label="Loss Latihan")
    plt.plot(__bahan_plottingan_model.loss_validasi, label="Loss Validasi")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss setiap Epoch")
    if not figure_sama:
        plt.show()

def plotting_latihan(figure_sama: bool = True):
    if not figure_sama:
        plt.figure()
    plt.plot(__bahan_plottingan_model.akurasi_latihan, label="Akurasi Latihan")
    plt.plot(__bahan_plottingan_model.akurasi_validasi, label="Akurasi Validasi")
    plt.xlabel("Epoch")
    plt.ylabel("Akurasi")
    plt.legend()
    plt.title("Akurasi setiap Epoch")
    if not figure_sama:
        plt.show()

def plotting_loss_dan_latih():
    __siapkan_figure()
    plotting_latihan()
    plotting_loss()
    plt.show()
