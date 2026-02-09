from typing import Final, Dict
import torch.nn as tnn

KLASIFIKASI_NAMA_JERUK: Final[Dict[int, str]] = {
    0: "gerga",
    1: "kalamansi"
}

class ModelCNNJeruk(tnn.Module):
    def __init__(self, jumlah_klasifikasi: int = 2):
        super(ModelCNNJeruk, self).__init__()    

        # siapkan layer ReLU: aktivasi layer
        self.relu = tnn.ReLU()
        # siapkan layer untuk di 'flatten'
        self.flatten = tnn.Flatten()

        # layer konvolusi pertama
        self.first_conv = tnn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        # -> pooling
        self.pool = tnn.MaxPool2d(kernel_size=2, stride=2)

        # layer konvolusi kedua
        self.second_conv = tnn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)

        # layer konvolusi ketiga
        self.third_conv = tnn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)

        # gabungan layer - layer
        self.fc1 = tnn.Linear(256, 128)
        self.fc2 = tnn.Linear(128, jumlah_klasifikasi)

    def forward(self, x):
        # pola conv -> relu -> pool
        x = self.pool(self.relu(self.first_conv(x)))
        x = self.pool(self.relu(self.second_conv(x)))
        x = self.pool(self.relu(self.third_conv(x)))

        # flatten semua
        x = self.flatten(x)

        # print(x.shape)
        # di aktifasi sebelum di kasih layer linear
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x
