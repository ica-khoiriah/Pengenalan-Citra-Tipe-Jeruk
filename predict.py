from local_const import KLASIFIKASI_NAMA_JERUK
from model_transforms import ModelTransform
from model_utils import perangkat_pilihan
from PIL import Image
import torch
from local_type import (
    Model,
    PrediksiSoftmax,
    PrediksiMax
)

def prediksi_softmax(model: Model, citra: Image) -> PrediksiSoftmax:
    citra_siap = ModelTransform.transform_validasi(citra)
    citra_siap = citra_siap.unsqueeze(0)
    hasil = None

    model.eval()

    perangkat_latih = torch.device(perangkat_pilihan())
    citra_siap.to(perangkat_latih)

    with torch.no_grad():
        output = model(citra_siap)
        hasil = torch.softmax(output, 1)

    return {
        KLASIFIKASI_NAMA_JERUK[0].title(): hasil[0][0],
        KLASIFIKASI_NAMA_JERUK[1].title(): hasil[0][1],
    }
    # return {
    #     KLASIFIKASI_NAMA_JERUK[i].title(): hasil[i].item()
    #     for i in range(len(hasil))
    # }

def prediksi_max(model: Model, citra: Image) -> PrediksiMax:
    citra_siap = ModelTransform.transform_validasi(citra)
    citra_siap = citra_siap.unsqueeze(0)

    model.eval()

    perangkat_latih = torch.device(perangkat_pilihan())
    citra_siap.to(perangkat_latih)

    with torch.no_grad():
        output = model(citra_siap)
        # nilai, potongan = torch.max(output, 1)
        hasil = torch.max(output, 1)

    # nama_kelas = potongan.item()
    # persentase_prediksi = nilai.item()

    # return {
    #     KLASIFIKASI_NAMA_JERUK[nama_kelas].title(): persentase_prediksi
    # }

    return {
        KLASIFIKASI_NAMA_JERUK[hasil.item()[1]].title(): hasil.item()[0],
    }

