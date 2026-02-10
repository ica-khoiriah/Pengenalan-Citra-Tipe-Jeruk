from plotting_utils import BahanPlottinganMatriksKonfusi
from sklearn.metrics import confusion_matrix
from local_type import KonfusiMatrix

def kalkulasi_konfusi_matrix(bpmk: BahanPlottinganMatriksKonfusi) -> KonfusiMatrix:
    return confusion_matrix(bpmk.label, bpmk.prediksi)