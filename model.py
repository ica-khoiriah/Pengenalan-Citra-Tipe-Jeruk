from local_const import TRANSFORM_SIZE
import torch.nn as tnn

class ModelCNNJeruk(tnn.Module):
    def __init__(self, jumlah_klasifikasi: int = 2):
        super(ModelCNNJeruk, self).__init__()    

        self.conv_kernel_size = 3
        self.conv_padding = 1

        # siapkan layer ReLU: aktivasi layer
        self.relu = tnn.ReLU()
        # siapkan layer untuk di 'flatten'
        self.flatten = tnn.Flatten()

        # layer konvolusi pertama
        self.first_conv = tnn.Conv2d(in_channels=3, out_channels=16, kernel_size=self.conv_kernel_size, padding=self.conv_padding)
        # -> pooling
        self.pool = tnn.MaxPool2d(kernel_size=2, stride=2)

        # layer konvolusi kedua
        self.second_conv = tnn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)

        # layer konvolusi ketiga
        self.third_conv = tnn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)

        perkalian_linear = (TRANSFORM_SIZE // 8) - self.conv_kernel_size + self.conv_padding
        self.linear_in_feature = TRANSFORM_SIZE * perkalian_linear * perkalian_linear
        self.linear_out_feature = 128

        # gabungan layer - layer
        self.fc1 = tnn.Linear(self.linear_in_feature, self.linear_out_feature)
        self.fc2 = tnn.Linear(self.linear_out_feature, jumlah_klasifikasi)

    def forward(self, x):
        # pola conv -> relu -> pool
        x = self.pool(self.relu(self.first_conv(x)))
        x = self.pool(self.relu(self.second_conv(x)))
        x = self.pool(self.relu(self.third_conv(x)))

        # print("before flatten", x.shape)

        # flatten semua
        x = self.flatten(x)
        # print("after flatten", x.shape)

        # print(x.shape)
        # di aktifasi sebelum di kasih layer linear
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x
