import torch
from torch import nn
from torch.nn import functional as F

import neural_net.extractors as extractors

# This code (both pspnet.py and extractors.py) is a modification of the one developed here https://github.com/Lextal/pspnet-pytorch

class PSPModule(nn.Module):
    """
    This is the core class of PSP (encoder part).
    it applies in parallel different pooling layers (1,2,3,6), each followed by 1x1 convolution.
    Their output is concatenated along depth, also adding the original features,
    and transformed again with a convolution.
    """

    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        """
        @param features: depth of the input tensor
        @param out_features: depth of the output tensor (i.e., number of filters)
        @param sizes: list of the different pooling kernel sizes you want to apply (Pyramid pooling)
        """
        super().__init__()
        self.stages = []
        # For each pyramid kernel size, create a stage (i.e., pyramid-pooling + convolution)
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        # This stage applies convolution to the concatenation of the results of pyramid kernels (stages)
        # concatenation of input tensors is made along the depth layer: features * (len(sizes) + 1)
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        """
        Applies down-pooling with specified kernel size, followed by 1x1 convolution
        @param features: number of input features = number of output features
        @param size: kernel size for pooling
        """
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size)) # Pooling with specified kernel size
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False) # 1x1 convolution
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        """
        Run the CNN module.
        @param feats: input tensor
        """
        h, w = feats.size(2), feats.size(3)

        # 1. Apply the stages with different pooling sizes
        # 2. upsample their output to the original with/height (bilinear)
        # 3. also use the original features (feats) as input for the next stage
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]

        # 1. Concatenate the stage outputs and the original features
        # 2. Apply convolution
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    """
    Base module for building the decoder
    """
    def __init__(self, in_channels, out_channels):
        """
        Create upsample module (size * 2).
        @param in_channels: depth of the input tensor
        @param out_channels: depth of the output tensor
        """
        super().__init__()
        self.conv = nn.Sequential(
            # 3x3 convolution
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        """
        @param x: input tensor
        """
        # Double the size of the input tensor with bilinear filter
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.interpolate(input=x, size=(h, w), mode='bilinear', align_corners=True)
        # Apply a convolutional layer
        return self.conv(p)


class PSPNet(nn.Module):
    """
    Pyramid scene parsing neural network.
    """
    def __init__(self, output_type, n_channels=3, regr_range=4, sizes=(1, 2, 3, 6), feature_size=512, backend='resnet34', pretrained=False):
        """
        @param output_type: 'regr' if your output is regression ([0,4] range). For classification, it should provide the number of classes.
        @param n_channels: number of input channels for the images
        @param regr_range: None for classification, int to specify output regression range -> [0, regr_range]
                           Regression output is Sigmoid(x)*regr_range
        @param sizes: list of pyramid pooling kernel sizes
        @param feature_size: depth of the feature extraction layer
        @param backend: name of the backend for feature extraction (e.g., resnet18, resnet34, resnet50, resnet101, resnet152)
        @param pretrained: True if you want to use pretrained backend
        """
        super().__init__()

        self.pretrained = pretrained
        self.regr_range = regr_range

        # 1. Select Feature extraction layer: from input image to features-tensor
        # pretrained: whether the feature extraction module is pretrained
        # input_depth: specifies the number of channels
        self.feats = getattr(extractors, backend)(pretrained, n_channels)

        # 2. Main module with the pyramid pooling layers (applied to extracted features)
#        psp_size = 2048 # output size of resnet --> TODO: qua era fissato in questo modo ma così da errore
#         psp_size = 512 # output size of resnet
        self.psp = PSPModule(feature_size, feature_size//2, sizes) # From psp_size to 1024 channels
        self.drop_1 = nn.Dropout2d(p=0.3)

        # 3. Add three upsample layers, with decreasing depth (from original 1024 to 64)
        self.up_1 = PSPUpsample(feature_size//2, feature_size//4)
        self.up_2 = PSPUpsample(feature_size//4, feature_size//8)
        self.up_3 = PSPUpsample(feature_size//8, feature_size//8)

        self.drop_2 = nn.Dropout2d(p=0.15)

        # 4. Final convolutional layer (1x1) from 64 depth to n_classes (output probabilities)
        # Regression or single-class classification -> use sigmoid
        self.output_type = output_type
        if output_type=='regr' or output_type==1:
            self.final = nn.Sequential(
                nn.Conv2d(feature_size//8, 1, kernel_size=1), # output depth is 1 (just regression)
                nn.Sigmoid()
            )
        # Multi class classification -> use softmax
        elif type(output_type) is int and self.output_type>1:
            self.final = nn.Sequential(
                nn.Conv2d(feature_size//8, output_type, kernel_size=1), # final number of classes
                # Applies softmax
                nn.LogSoftmax()
            )


        # TODO: questo mi sembra inutile, sembra un "baseline" classifier per confrontarlo con la pspnet
        # self.classifier = nn.Sequential(
        #     nn.Linear(deep_features_size, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, n_classes)
        # )

    def forward(self, x):
        f, class_f = self.feats(x) # Extract the 2048 features
        p = self.psp(f)            # Apply pyramid modules
        p = self.drop_1(p)         # dropout

        p = self.up_1(p)           # upsampling1
        p = self.drop_2(p)

        p = self.up_2(p)           # upsampling2
        p = self.drop_2(p)

        p = self.up_3(p)           # upsampling3
        p = self.drop_2(p)

        auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))

        # final convolution
        if self.output_type=='regr':  # Regression for Satellite images [0, regr_range]
            return self.regr_range * self.final(p)
        else:
            return self.final(p)      # Softmax/sigmoid output (classification)

        # TODO: ho rimosso il classificatore 2 che secondo me e' inutile
        # return self.final(p), self.classifier(auxiliary)

    def initialize_weights(self, seed):
        """
        Initialization of model weights ()
        """
        pass