import torch
import torch.nn.functional as F

from torch import nn
from .unet_parts import *
from neural_net.segnet import SegNet
from collections import OrderedDict

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, act='relu', first_ch_out=64, alpha=1.0, dropout=True, gradcam=False, end_withsigm = False):
        super(UNet, self).__init__()
        self.first_ch_out = first_ch_out
        self.n_classes = n_classes
        self.act = act
        self.alpha = alpha
        self.gradcam = gradcam
        self.end_withsigm = end_withsigm
        self.gradients = None
        
        self.inc = inconv(n_channels, first_ch_out)
        self.down1 = down(first_ch_out, first_ch_out * 2, act=act, alpha=alpha)
        self.down2 = down(first_ch_out * 2, first_ch_out * 4, act=act, alpha=alpha)
        self.down3 = down(first_ch_out * 4, first_ch_out * 8, act=act, alpha=alpha)
        self.down4 = down(first_ch_out * 8, first_ch_out * 16, act=act, alpha=alpha)
        self.up1 = up(first_ch_out * 16, first_ch_out * 8, act=act, alpha=alpha, dropout=dropout)
        self.up2 = up(first_ch_out * 8, first_ch_out * 4, act=act, alpha=alpha, dropout=dropout)
        self.up3 = up(first_ch_out * 4, first_ch_out * 2, act=act, alpha=alpha, dropout=dropout)
        self.up4 = up(first_ch_out * 2, first_ch_out, act=act, alpha=alpha, dropout=dropout)
        self.outc = outconv(first_ch_out, n_classes)
        if self.end_withsigm: self.sigmoid = nn.Sigmoid() # TODO: softmax for multiple classes

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        if self.gradcam:
            x.register_hook(self.hook_func)

        x = self.outc(x)
        if self.end_withsigm:
            x = self.sigmoid(x)
        return x

    def hook_func(self, grad):
        self.gradients = grad

    def get_gradients(self):
        return self.gradients

    def get_activation(self, x):
        x1 = self.inc(x)    # input -> 64
        x2 = self.down1(x1) # 64 -> 128
        x3 = self.down2(x2) # 128 -> 256
        x4 = self.down3(x3) # 256 -> 512
        x5 = self.down4(x4) # 512 -> 1024
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        if self.end_withsigm:
            x = self.sigmoid(x)
        return x

class ConcatenatedUNet(nn.Module):
    def __init__(self, n_channels: int, act: str="relu", first_ch_out: int=64, alpha: float=1.0, dropout: bool=True, bin_channels=1, threshold: float=0.5, gradcam=0):
        super(ConcatenatedUNet, self).__init__()
        self.n_channels = n_channels
        self.act = act
        self.first_ch_out = first_ch_out
        self.alpha = alpha
        self.dropout = dropout
        self.bin_channels = bin_channels
        self.threshold = threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gradcam = [False, False]
        self.grad_module = None
        if gradcam > 0: self.gradcam[gradcam-1] = True
        self.binary_unet = UNet(n_channels, bin_channels, act, first_ch_out, alpha, dropout, gradcam=self.gradcam[0])
        self.sigm = nn.Sigmoid()
        self.regression_unet = UNet(n_channels, 1, act, first_ch_out, alpha, dropout, gradcam=self.gradcam[1])
        if self.gradcam[0]: self.grad_module = self.binary_unet
        if self.gradcam[1]: self.grad_module = self.regression_unet
        return

    def forward(self, x):
        binary_out = self.binary_unet(x)
        if self.bin_channels == 1:
            tmp = self.sigm(binary_out)
            multiplier = torch.where(tmp > self.threshold, torch.tensor(1.0, device=self.device), torch.tensor(0.0, device=self.device))
        else:
            tmp = binary_out.argmax(axis=1, keepdim=True)
            multiplier = tmp
        regr_input = x * multiplier
        regr_output = self.regression_unet(regr_input)
        return binary_out, regr_output

    def get_activation(self, x):
        if self.gradcam[0]: return self.grad_module.get_activation(x[0])

        binary_out = self.binary_unet(x)
        if self.bin_channels == 1:
            tmp = self.sigm(binary_out)
            multiplier = torch.where(tmp > self.threshold, torch.tensor(1.0, device=self.device),
                                     torch.tensor(0.0, device=self.device))
        else:
            tmp = binary_out.argmax(axis=1, keepdim=True)
            multiplier = tmp
        regr_input = x * multiplier

        if self.gradcam[1]: return self.regression_unet.get_activation(regr_input)

    def get_gradients(self):
        return self.grad_module.get_gradients()

    @classmethod
    def _set_model_grad_flag(cls, model, flag: bool):
        for param in model.parameters():
            param.requires_grad = flag

    @classmethod
    def _freeze_model(cls, model):
        cls._set_model_grad_flag(model, False)

    @classmethod
    def _unfreeze_model(cls, model):
        cls._set_model_grad_flag(model, True)

    def freeze_binary_unet(self):
        self._freeze_model(self.binary_unet)

    def freeze_regression_unet(self):
        self._freeze_model(self.regression_unet)

    def unfreeze_binary_unet(self):
        self._unfreeze_model(self.binary_unet)

    def unfreeze_regression_unet(self):
        self._unfreeze_model(self.regression_unet)


class MixedNet(nn.Module):
    def __init__(self, n_channels: int, first_ch_out: int=64, bin_net=UNet, regr_net= SegNet, alpha: float=1.0, dropout: bool=True, bin_channels=1, threshold: float=0.5):
        super(MixedNet, self).__init__()
        self.n_channels = n_channels
        self.first_ch_out = first_ch_out
        self.alpha = alpha
        self.dropout = dropout
        self.bin_channels = bin_channels
        self.threshold = threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.binary_unet = bin_net(n_channels, n_classes=bin_channels, first_ch_out=first_ch_out)
        self.sigm = nn.Sigmoid()
        self.regression_unet = regr_net(n_channels, n_classes=1, first_ch_out=first_ch_out)
        return

    def forward(self, x):
        binary_out = self.binary_unet(x)
        if self.bin_channels == 1:
            tmp = self.sigm(binary_out)
            multiplier = torch.where(tmp > self.threshold, torch.tensor(1.0, device=self.device), torch.tensor(0.0, device=self.device))
        else:
            tmp = binary_out.argmax(axis=1, keepdim=True)
            multiplier = tmp
        regr_input = x * multiplier
        regr_output = self.regression_unet(regr_input)
        return binary_out, regr_output

    @classmethod
    def _set_model_grad_flag(cls, model, flag: bool):
        for param in model.parameters():
            param.requires_grad = flag

    @classmethod
    def _freeze_model(cls, model):
        cls._set_model_grad_flag(model, False)

    @classmethod
    def _unfreeze_model(cls, model):
        cls._set_model_grad_flag(model, True)

    def freeze_binary_unet(self):
        self._freeze_model(self.binary_unet)

    def freeze_regression_unet(self):
        self._freeze_model(self.regression_unet)

    def unfreeze_binary_unet(self):
        self._unfreeze_model(self.binary_unet)

    def unfreeze_regression_unet(self):
        self._unfreeze_model(self.regression_unet)


class ANN(nn.Module):
    def __init__(self, in_features, hidden_layers, bn=False, dropout=False, last_dropout=True, drop_prob=0.5):
        super(ANN, self).__init__()
        self.in_features = in_features
        self.hidden_layers = hidden_layers
        self.bn = bn
        self.dropout = dropout
        self.drop_prob = drop_prob

        hl = OrderedDict()
        
        l = nn.Linear(in_features, hidden_layers[0])
        hl['in'] = l
        out_features = hidden_layers[0]
        for idx, layer in enumerate(hidden_layers):
            in_features = layer
            out_features = hidden_layers[idx + 1] if idx + 1 < len(hidden_layers) else 1
            l = nn.Linear(in_features, out_features)

            if last_dropout and idx == len(hidden_layers) - 1:
                hl['last_dropout'] = nn.Dropout(drop_prob)

            hl['fc%d' % idx] = l
            if idx < len(hidden_layers) - 1:
                if bn:
                    hl['bn%d' % idx] = nn.BatchNorm1d(out_features)
                if dropout:
                    hl['drop%d' % idx] = nn.Dropout(drop_prob)

        self.ann = nn.Sequential(hl)

    def forward(self, x):
        return self.ann(x)

class UNetConcANN(nn.Module):
    def __init__(self, n_channels: int, act: str, hidden_layers: list,first_ch_out: int=64, alpha: float=1.0, dropout: bool=True, threshold: float=0.5):
        super(UNetConcANN, self).__init__()
        self.n_channels = n_channels
        self.act = act
        self.first_ch_out = first_ch_out
        self.alpha = alpha
        self.dropout = dropout
        self.threshold = threshold
        self.hidden_layers = hidden_layers

        self.unet = UNet(n_channels, 1, act, first_ch_out, alpha, dropout)
        self.sigm = nn.Sigmoid()
        
        self.ann = ANN(n_channels, hidden_layers, False, False)

    def forward(self, x):
        x = self.unet(x)
        bin_out = self.sigm(x)
        x = self.ann(bin_out)
        return x


class UNetANN(nn.Module):
    def __init__(self, n_channels: int, act: str, ann: list, first_ch_out: int=64, alpha: float=1.0, dropout: bool=True):
        super(UNetANN, self).__init__()
        self.n_channels = n_channels
        self.act = act
        self.first_ch_out = first_ch_out
        self.alpha = alpha

        self.inc = inconv(n_channels, first_ch_out)
        self.down1 = down(first_ch_out, first_ch_out * 2, act=act, alpha=alpha)
        self.down2 = down(first_ch_out * 2, first_ch_out * 4, act=act, alpha=alpha)
        self.down3 = down(first_ch_out * 4, first_ch_out * 8, act=act, alpha=alpha)
        self.down4 = down(first_ch_out * 8, first_ch_out * 16, act=act, alpha=alpha)
        self.up1 = up(first_ch_out * 16, first_ch_out * 8, act=act, alpha=alpha, dropout=dropout)
        self.up2 = up(first_ch_out * 8, first_ch_out * 4, act=act, alpha=alpha, dropout=dropout)
        self.up3 = up(first_ch_out * 4, first_ch_out * 2, act=act, alpha=alpha, dropout=dropout)
        self.up4 = up(first_ch_out * 2, first_ch_out, act=act, alpha=alpha, dropout=dropout)
        
        self.ann = ANN(first_ch_out, ann)
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        shape = x.shape

        x = x.permute((0, 2, 3, 1)).reshape((-1, x.shape[1]))
        x = self.ann(x)
        x = x.reshape(shape[0], 1, shape[2], shape[3])
        return x
