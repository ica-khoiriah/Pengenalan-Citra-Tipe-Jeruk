from model_utils import muat_model
from model import ModelCNNJeruk
from textwrap import dedent
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
    css_citra_input = dedent("""
    #kotak-citra-input img {
        max-height: 300px;
        object-fit: contain;
    }
    """)

    with gr.Blocks() as app:
        with gr.Row():
            with gr.Column(scale=1):
                citra_input = gr.Image(type="pil", elem_id="kotak-citra-input", height=300, label="Input Citra")

                with gr.Row():
                    tombol_prediksi = gr.Button("Prediksi", variant="primary")
                    tombol_clear = gr.Button("Clear")
            
            with gr.Column(scale=1):
                prediksi_output = gr.Label(label="Output Prediksi")
        
        tombol_prediksi.click(
            fungsi_prediksi,
            inputs=citra_input,
            outputs=prediksi_output
        )
        tombol_clear.click(
            fn=lambda: (None, None),
            inputs=[],
            outputs=[citra_input, prediksi_output]
        )

    app_modifikasi = partial(app.launch, css=css_citra_input)


    # with gr.Blocks(css=css_citra_input) as app:
    #     citra_input = gr.Image(type="pil", elem_id="kotak-citra-input")
    #     tombol_prediksi = gr.Button("Prediksi")
    #     tombol_clear = gr.Button("Clear")
    #     prediksi_output = gr.Label()

    #     tombol_prediksi.click(
    #         fungsi_prediksi,
    #         inputs=citra_input,
    #         outputs=prediksi_output
    #     )
    #     tombol_clear.click(
    #         fn=lambda: (None, None),
    #         inputs=[],
    #         outputs=[citra_input, prediksi_output]
    #     )

    # app = gr.Interface(
    #     fn=fungsi_prediksi,
    #     inputs=gr.Image(type='pil'),
    #     outputs=gr.Label(num_top_classes=2)
    # )

    return app_modifikasi

