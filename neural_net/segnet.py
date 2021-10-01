import torch
import torch.nn as nn
import torch.nn.functional as F


class SegNet(nn.Module):
    """SegNet: A Deep Convolutional Encoder-Decoder Architecture for
    Image Segmentation. https://arxiv.org/abs/1511.00561
    See https://github.com/alexgkendall/SegNet-Tutorial for original models.
    Args:
        n_classes (int): number of classes to segment
        n_channels (int): number of input features in the fist convolution
        drop_rate (float): dropout rate of each encoder/decoder module
        filter_config (list of 5 ints): number of output features at each level
    """


    def __init__(self, n_channels: int, n_classes=1, drop_rate=0.5, first_ch_out=64):
        super(SegNet, self).__init__()

        filter_config = (first_ch_out, 2*first_ch_out, 4*first_ch_out, 8*first_ch_out, 8*first_ch_out)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        # setup number of conv-bn-relu blocks per module and number of filters
        encoder_n_layers = (2, 2, 3, 3, 3)
        encoder_filter_config = (n_channels,) + filter_config
        decoder_n_layers = (3, 3, 3, 2, 1)
        decoder_filter_config = filter_config[::-1] + (filter_config[0],)

        for i in range(0, 5):
            # encoder architecture
            self.encoders.append(_Encoder(encoder_filter_config[i],
                                          encoder_filter_config[i + 1],
                                          encoder_n_layers[i], drop_rate))

            # decoder architecture
            self.decoders.append(_Decoder(decoder_filter_config[i],
                                          decoder_filter_config[i + 1],
                                          decoder_n_layers[i], drop_rate))

        # final classifier (equivalent to a fully connected layer)
        self.classifier = nn.Conv2d(filter_config[0], n_classes, 3, 1, 1)

    def forward(self, x):
        indices = []
        unpool_sizes = []
        feat = x

        # encoder path, keep track of pooling indices and features size
        for i in range(0, 5):
            (feat, ind), size = self.encoders[i](feat)
            indices.append(ind)
            unpool_sizes.append(size)

        # decoder path, upsampling with corresponding indices and size
        for i in range(0, 5):
            feat = self.decoders[i](feat, indices[4 - i], unpool_sizes[4 - i])

        return self.classifier(feat)


class _Encoder(nn.Module):
    def __init__(self, n_in_feat, n_out_feat, n_blocks=2, drop_rate=0.5):
        """Encoder layer follows VGG rules + keeps pooling indices
        Args:
            n_in_feat (int): number of input features
            n_out_feat (int): number of output features
            n_blocks (int): number of conv-batch-relu block inside the encoder
            drop_rate (float): dropout rate to use
        """
        super(_Encoder, self).__init__()

        layers = [nn.Conv2d(n_in_feat, n_out_feat, 3, 1, 1),
                  nn.BatchNorm2d(n_out_feat),
                  nn.ReLU(inplace=True)]

        if n_blocks > 1:
            layers += [nn.Conv2d(n_out_feat, n_out_feat, 3, 1, 1),
                       nn.BatchNorm2d(n_out_feat),
                       nn.ReLU(inplace=True)]
            if n_blocks == 3:
                layers += [nn.Dropout(drop_rate)]

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        output = self.features(x)
        return F.max_pool2d(output, 2, 2, return_indices=True), output.size()


class ConcatenatedSegNet(nn.Module):
    def __init__(self, n_channels: int, first_ch_out: int=64, alpha: float=1.0, dropout: bool=True, bin_channels=1, threshold: float=0.5):
        super(ConcatenatedSegNet, self).__init__()
        self.n_channels = n_channels
        self.first_ch_out = first_ch_out
        self.alpha = alpha
        self.dropout = dropout
        self.bin_channels = bin_channels
        self.threshold = threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.binary_unet = SegNet(n_channels, n_classes=bin_channels)
        self.sigm = nn.Sigmoid()
        self.regression_unet = SegNet(n_channels, n_classes=1)
        return

    def forward(self, x):
        binary_out = self.binary_unet(x)
        if self.bin_channels == 1:
            tmp = self.sigm(binary_out)
            multiplier = torch.where(tmp > self.threshold, torch.tensor(1.0, device=self.device),
                                     torch.tensor(0.0, device=self.device))
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


class _Decoder(nn.Module):
    """Decoder layer decodes the features by unpooling with respect to
    the pooling indices of the corresponding decoder part.
    Args:
        n_in_feat (int): number of input features
        n_out_feat (int): number of output features
        n_blocks (int): number of conv-batch-relu block inside the decoder
        drop_rate (float): dropout rate to use
    """
    def __init__(self, n_in_feat, n_out_feat, n_blocks=2, drop_rate=0.5):
        super(_Decoder, self).__init__()

        layers = [nn.Conv2d(n_in_feat, n_in_feat, 3, 1, 1),
                  nn.BatchNorm2d(n_in_feat),
                  nn.ReLU(inplace=True)]

        if n_blocks > 1:
            layers += [nn.Conv2d(n_in_feat, n_out_feat, 3, 1, 1),
                       nn.BatchNorm2d(n_out_feat),
                       nn.ReLU(inplace=True)]
            if n_blocks == 3:
                layers += [nn.Dropout(drop_rate)]

        self.features = nn.Sequential(*layers)

    def forward(self, x, indices, size):
        unpooled = F.max_unpool2d(x, indices, 2, 2, 0, size)
        return self.features(unpooled)