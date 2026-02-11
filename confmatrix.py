from plotting_utils import BahanPlottinganMatriksKonfusi
from sklearn.metrics import confusion_matrix
from local_type import KonfusiMatrix

def kalkulasi_konfusi_matrix(bahan_matriks_konfusi: BahanPlottinganMatriksKonfusi) -> KonfusiMatrix:
    return confusion_matrix(
        bahan_matriks_konfusi.label,
        bahan_matriks_konfusi.prediksi
    )
