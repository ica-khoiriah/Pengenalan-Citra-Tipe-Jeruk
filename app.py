from model_utils import muat_model
from model import ModelCNNJeruk
from local_type import UIGradio
from functools import partial
from pathlib import Path
import gradio as gr
from predict import (
    prediksi_softmax
)

def ready(path_model: Path) -> UIGradio:
    model = ModelCNNJeruk(jumlah_klasifikasi=2)

    muat_model(model, path_model)

    fungsi_prediksi = partial(prediksi_softmax, model)

    app = gr.Interface(
        fn=fungsi_prediksi,
        inputs=gr.Image(type='pil'),
        outputs=gr.Label(num_top_classes=2)
    )

    return app

