from torch import nn
import torch

from neural_net.utils import eval_object

from neural_net.unet import UNet
from neural_net.attention_unet import AttentionUnet
from neural_net.canet_parts.networks.network import Comprehensive_Atten_Unet


class ConcatenatedModel(nn.Module):
    def __init__(self, model_dict, threshold: float=0.5):
        super(ConcatenatedModel, self).__init__()
        self.binary_unet = eval_object(model_dict)
        
        self.backbone = model_dict.name
        self.sigm = nn.Sigmoid()
        self.regression_unet = eval_object(model_dict)
        self.threshold = threshold

    def forward(self, x):
        binary_out = self.binary_unet(x)
        tmp = self.sigm(binary_out)
        multiplier = (tmp > self.threshold).to(tmp)
        
        regr_input = x * multiplier
        regr_output = self.regression_unet(regr_input)
        return binary_out, regr_output
    
    def get_attention_maps(self, x):
        if hasattr(self.binary_unet, "get_attention_maps"):
            bin_attentions = self.binary_unet.get_attention_maps(x)
            regr_attentions = self.regression_unet.get_attention_maps(x)
        return bin_attentions, regr_attentions

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
