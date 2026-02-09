from model import KLASIFIKASI_NAMA_JERUK
from torchvision import transforms
from PIL import Image as pimg
from typing import Final, Any
from typing import Callable
from enum import StrEnum
from pathlib import Path
import torch.nn as tnn
from PIL import Image
import torch

TRANSFORM: Final[transforms.Compose] = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

__model: tnn.Module | None = None

def set_model(model: tnn.Module):
    global __model
    __model = model

class TipePrediksi(StrEnum):
    Max = "max"
    Softmax = "softmax"

def prediksi_citra[T: Any](image: Image, fungsi_prediksi: Callable[[T], T]) -> torch.Tensor:
    citra = TRANSFORM(image)
    citra = citra.unsqueeze(0)

    __model.eval()

    with torch.no_grad():
        output = __model(citra)
        result = fungsi_prediksi(output, 1)

    return result

def prediksi_softmax(image: Image) -> torch.Tensor:
    return prediksi_citra(image, torch.softmax)

def prediksi_max(image: Image) -> torch.Tensor:
    return prediksi_citra(image, torch.max)

def prediksi_softmax_gr(image: Image) -> torch.Tensor:
    pred = prediksi_citra(image, torch.softmax)
    return {
        KLASIFIKASI_NAMA_JERUK[0].title(): pred[0][0],
        KLASIFIKASI_NAMA_JERUK[1].title(): pred[0][1],
    }

def prediksi_max_gr(image: Image) -> torch.Tensor:
    pred = prediksi_citra(image, torch.max)
    return {
        KLASIFIKASI_NAMA_JERUK[pred.item()[1]]: pred.item()[0]
    }

def prediksi_citra_path(path_citra: Path, *, tipe_prediksi: TipePrediksi = TipePrediksi.Softmax) -> torch.Tensor:
    TIPE_WARNA: Final[str] = "RGB"

    citra = pimg.open(path_citra).convert(TIPE_WARNA)

    if (tipe_prediksi == TipePrediksi.Max):
        return prediksi_max(citra)
    elif (tipe_prediksi == TipePrediksi.Softmax):
        return prediksi_softmax(citra)
    else:
        raise ValueError("'tipe_prediksi' argument is incorrect; expected type of 'TipePrediksi'.")

def prediksi_citra_gr(*args, **kwargs) -> torch.Tensor:
    result: torch.Tensor = prediksi_citra_path(*args, **kwargs)
    if kwargs['tipe_prediksi'] == TipePrediksi.Max:
        return {
            KLASIFIKASI_NAMA_JERUK[result.item()[1]]: result.item()[0]
        }
    elif kwargs['tipe_prediksi'] == TipePrediksi.Softmax:
        return {
            KLASIFIKASI_NAMA_JERUK[0].title(): result.item()[0][0],
            KLASIFIKASI_NAMA_JERUK[1].title(): result.item()[0][1]
        }