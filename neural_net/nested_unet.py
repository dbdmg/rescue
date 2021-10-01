from torch import nn
import torch

from neural_net.unet_parts import double_conv




class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class NestedUNet(nn.Module):

    def __init__(self, output_type, n_channels, act='relu', regr_range=4, first_ch_out=32, alpha=1.0, dropout=True, gradcam=False, deep_supervision=False):
        """
        @param output_type: 'regr' if your output is regression. For classification, it should provide the number of classes.
        @param n_channels: number of input channels for the images
        @param act: activation function after convolutional layers ('relu' or 'elu')
        @param regr_range: None for classification, int to specify output regression range -> [0, regr_range]
                           Regression output is Sigmoid(x)*regr_range
        @param first_ch_out: size of extracted features; 64 to be coherent with normal UNet, 32 in original NestedUnet
        """
        super().__init__()

        nb_filter = [first_ch_out, first_ch_out*2, first_ch_out*4, first_ch_out*8, first_ch_out*16]
        self.dropout = dropout
        if self.dropout:
            self.drop_layer = nn.Dropout(0.25)
        self.deep_supervision = deep_supervision
        self.output_type = output_type
        self.regr_range = regr_range

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = double_conv(n_channels, nb_filter[0], act=act, alpha=alpha)   # in -> 32/64
        self.conv1_0 = double_conv(nb_filter[0], nb_filter[1], act=act, alpha=alpha) # -> 64/128
        self.conv2_0 = double_conv(nb_filter[1], nb_filter[2], act=act, alpha=alpha) # -> 128/256
        self.conv3_0 = double_conv(nb_filter[2], nb_filter[3], act=act, alpha=alpha) # -> 256/512
        self.conv4_0 = double_conv(nb_filter[3], nb_filter[4], act=act, alpha=alpha) # -> 512/1024

        self.conv0_1 = double_conv(nb_filter[0]+nb_filter[1], nb_filter[0], act=act, alpha=alpha)
        self.conv1_1 = double_conv(nb_filter[1]+nb_filter[2], nb_filter[1], act=act, alpha=alpha)
        self.conv2_1 = double_conv(nb_filter[2]+nb_filter[3], nb_filter[2], act=act, alpha=alpha)
        self.conv3_1 = double_conv(nb_filter[3]+nb_filter[4], nb_filter[3], act=act, alpha=alpha)

        self.conv0_2 = double_conv(nb_filter[0]*2+nb_filter[1], nb_filter[0], act=act, alpha=alpha)
        self.conv1_2 = double_conv(nb_filter[1]*2+nb_filter[2], nb_filter[1], act=act, alpha=alpha)
        self.conv2_2 = double_conv(nb_filter[2]*2+nb_filter[3], nb_filter[2], act=act, alpha=alpha)

        self.conv0_3 = double_conv(nb_filter[0]*3+nb_filter[1], nb_filter[0], act=act, alpha=alpha)
        self.conv1_3 = double_conv(nb_filter[1]*3+nb_filter[2], nb_filter[1], act=act, alpha=alpha)

        self.conv0_4 = double_conv(nb_filter[0]*4+nb_filter[1], nb_filter[0], act=act, alpha=alpha)   
    
        
        if self.output_type == 'regr':
            n_classes = 1
        elif type(self.output_type) is int:
            n_classes = self.output_type  # Classification

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)

    def cat_dout(self, tensors):
        """Concatenation and dropout"""
        cat = torch.cat(tensors, 1)
        if self.dropout:
            cat = self.drop_layer(cat)
        return cat

    def forward(self, input):
        # x0_0 = self.conv0_0(input)
        # x1_0 = self.conv1_0(self.pool(x0_0))
        # x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        #
        # x2_0 = self.conv2_0(self.pool(x1_0))
        # x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        # x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        #
        # x3_0 = self.conv3_0(self.pool(x2_0))
        # x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        # x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        # x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        #
        # x4_0 = self.conv4_0(self.pool(x3_0))
        # x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        # x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        # x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        # x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(self.cat_dout([x0_0, self.up(x1_0)]))
        
        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(self.cat_dout([x1_0, self.up(x2_0)]))
        x0_2 = self.conv0_2(self.cat_dout([x0_0, x0_1, self.up(x1_1)]))
        
        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(self.cat_dout([x2_0, self.up(x3_0)]))
        x1_2 = self.conv1_2(self.cat_dout([x1_0, x1_1, self.up(x2_1)]))
        x0_3 = self.conv0_3(self.cat_dout([x0_0, x0_1, x0_2, self.up(x1_2)]))
        
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(self.cat_dout([x3_0, self.up(x4_0)]))
        x2_2 = self.conv2_2(self.cat_dout([x2_0, x2_1, self.up(x3_1)]))
        x1_3 = self.conv1_3(self.cat_dout([x1_0, x1_1, x1_2, self.up(x2_2)]))
        x0_4 = self.conv0_4(self.cat_dout([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)]))
        if self.deep_supervision:
            pass
            print("Deep supervision not implemented yet...")
            exit()
            # output1 = self.final1(x0_1)
            # output2 = self.final2(x0_2)
            # output3 = self.final3(x0_3)
            # output4 = self.final4(x0_4)
            # return [output1, output2, output3, output4]

        else:
            # Regression vs classification
            output = self.final(x0_4)
            if self.output_type == 'regr':
                # return self.regr_range*torch.sigmoid(output) # Regression for Satellite images [0, regr_range]
                return output
            elif type(self.output_type) is int and self.output_type>1:
                return nn.LogSoftmax(output)   # Softmax output (classification)
            else:                              # Single class output -> use sigmoid
                # return nn.Sigmoid()(output)
                return output # TODO: Sigmoid should be already present in the step after

class ConcatenatedNestedUNet(nn.Module):
    def __init__(self, n_channels: int, act: str, first_ch_out: int=32, alpha: float=1.0, dropout: bool=True, bin_channels=1, threshold: float=0.5):
        super(ConcatenatedNestedUNet, self).__init__()
        self.n_channels = n_channels
        self.act = act
        self.first_ch_out = first_ch_out
        self.alpha = alpha
        self.dropout = dropout
        self.bin_channels = bin_channels
        self.threshold = threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.binary_unet = NestedUNet(bin_channels, n_channels=n_channels, act=act, first_ch_out=first_ch_out, alpha=alpha, dropout=dropout)
        self.sigm = nn.Sigmoid()
        self.regression_unet = NestedUNet('regr', n_channels=n_channels, act=act, first_ch_out=first_ch_out, alpha=alpha, dropout=dropout)
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
