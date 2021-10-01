import torch

from torch import nn

def _compute_shape(dim, padding, kernel_size, stride, dilation=1):
    num = dim + (2 * padding) - dilation * (kernel_size - 1) - 1
    result = (num / stride) + 1
    assert result > 0
    if isinstance(result, float):
        assert result.is_integer()
    return result

def compute_conv_out_shape(dim, padding, kernel_size, stride, dilation=1):
    return _compute_shape(dim, padding, kernel_size, stride, dilation=dilation)

def compute_maxpool_out_shape(dim, padding, kernel_size, stride, dilation=1):
    return _compute_shape(dim, padding, kernel_size, stride, dilation=dilation)

def compute_avg_pool_shape(dim, padding, kernel_size, stride):
    return _compute_shape(dim, padding, kernel_size=(kernel_size + 1), stride=stride, dilation=1)

class CNNModule(nn.Module):
    def __init__(self, window_size: int, in_features: int, cnn_list: list, pool_type='max'):
        super(CNNModule, self).__init__()
        self.window_size = window_size
        self.in_features = in_features
        self.cnn_list = cnn_list
        self.pool_type = pool_type

        assert pool_type == 'max' or pool_type == 'avg'

        cnn = []
        in_shape = in_features
        for idx, out_shape in enumerate(cnn_list):
            cnn.append(nn.Conv2d(in_shape, out_shape, kernel_size=3))
            cnn.append(nn.BatchNorm2d(out_shape))
            cnn.append(nn.ReLU())
            if pool_type == 'max':
                pool = nn.MaxPool2d(3, stride=1)
            elif pool_type == 'avg':
                pool = nn.AvgPool2d(kernel_size=3, stride=1)
            in_shape = out_shape
        self.cnn = nn.Sequential(cnn)

        return

    def compute_output_shape(self):
        dim = self.window_size
        for idx, out_shape in enumerate(cnn_list):
            dim = compute_conv_out_shape(dim, padding=0, kernel_size=3, stride=1, dilation=1)
            if self.pool_type == 'max':
                dim = compute_maxpool_out_shape(dim, padding=0, kernel_size=3, stride=1, dilation=1)
            elif self.pool_type == 'avg':
                dim = compute_avg_pool_shape(dim, padding=0, kernel_size=3, stride=1, dilation=1)
            n_channels = out_shape
        assert n_channels == self.cnn_list[-1]
        return dim, n_channels

    def forward(self, x):
        x = self.cnn(x)
        return x

class FCModule(nn.Module):
    def __init__(self, input_dim: int, fc_list: list, n_classes: int):
        super(FCModule, self).__init__()
        self.input_dim = input_dim
        self.fc_list = fc_list
        self.n_classes = n_classes

        fc = []
        in_shape = input_dim
        for idx, neurons in enumerate(fc_list):
            fc.append(nn.Linear(in_shape, neurons))
            fc.append(nn.ReLU())
            in_shape = neurons
        fc.append(nn.Dropout(0.5))
        fc.append(nn.Linear(in_shape, n_classes))
        self.fc = nn.Sequential(fc)
        return

    def forward(self, x):
        x = self.fc(x)
        return x

class ContextCNN(nn.Module):
    def __init__(self, window_size: int, in_features: int, cnn_list: list, fc_list: list, pool_type: str, n_classes: int):
        super(ContextCNN, self).__init__()
        self.window_size = window_size
        self.fc_list = fc_list
        self.pool_type = pool_type
        self.n_classes = n_classes

        self.cnn = CNNModule(window_size, in_features, cnn_list, pool_type)
        dim, n_channels = self.cnn.compute_output_shape()
        n_features = dim * dim * n_channels
        self.fc = FCModule(n_features, fc_list, n_classes)
        return

    def forward(self, x):
        x = self.cnn(x)
        n_samples = x.size(0)
        x = x.view(n_samples, -1)
        x = self.fc(x)
        return x