from torch.utils.data import DataLoader
from matplotlib.figure import Figure
from torchvision import transforms
from matplotlib.axes import Axes
import torch.nn as tnn
import gradio as gr
from numpy import ndarray
import torch
from typing import (
    TypedDict,
    Tuple,
    Dict,
)

class PrediksiSoftmax(TypedDict):
    Gerga: HasilPrediksi
    Kalamansi: HasilPrediksi

type Plottingan = Tuple[float]
type TipeCompose = transforms.Compose
type NamaPerangkat = str
type KonfusiMatrix = ndarray
type DataLoaderLatih = DataLoader
type DataLoaderValidasi = DataLoader
type Model = tnn.Module
type Tensor = torch.Tensor
type HasilPrediksi = Tensor|int|float
type PrediksiMax = Dict[str, HasilPrediksi]
type UIGradio = gr.Interface
type MatplotAxes = Axes
type MatplotFigure = Figure