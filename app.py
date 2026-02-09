from predict import prediksi_softmax_gr, set_model
from model_utils import muat_model
from model import ModelCNNJeruk
from pathlib import Path
import gradio as gr

def ready() -> gr.Interface:
    model = ModelCNNJeruk(jumlah_klasifikasi=2)
    muat_model(model, Path("./model/model-09_02_26_v2_13.pth"))
    set_model(model)
    model.eval()

    app = gr.Interface(
        fn=prediksi_softmax_gr,
        inputs=gr.Image(type='pil'),
        outputs=gr.Label(num_top_classes=2)
    )

    return app

