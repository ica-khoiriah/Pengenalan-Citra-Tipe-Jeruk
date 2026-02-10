from local_const import KLASIFIKASI_NAMA_JERUK
from model_transforms import ModelTransform
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
    hasil: None | torch.Tensor = None

    model.eval()

    with torch.no_grad():
        output = model(citra_siap)
        hasil = torch.softmax(output, 1)

    return {
        KLASIFIKASI_NAMA_JERUK[0].title(): hasil[0][0],
        KLASIFIKASI_NAMA_JERUK[1].title(): hasil[0][1],
    }

def prediksi_max(model: Model, citra: Image) -> PrediksiMax:
    citra_siap = ModelTransform.transform_validasi(citra)
    citra_siap = citra_siap.unsqueeze(0)
    hasil: None | torch.Tensor = None

    model.eval()

    with torch.no_grad():
        output = model(citra_siap)
        hasil = torch.max(output, 1)

    return {
        KLASIFIKASI_NAMA_JERUK[hasil.item()[1]].title(): hasil.item()[0],
    }
