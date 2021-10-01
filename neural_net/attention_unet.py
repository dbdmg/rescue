import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.base import modules as md

from typing import Optional, Union, List
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base.model import SegmentationModel
from segmentation_models_pytorch.base.heads import SegmentationHead, ClassificationHead

from neural_net.canet_parts.layers.grid_attention_layer import MultiAttentionBlock


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
            use_attention_layer=False,
    ):
        super().__init__()
        self.use_attention_layer = use_attention_layer
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)
        if skip_channels != 0:
            self.attention_layer = MultiAttentionBlock(
                in_size=in_channels,
                gate_size=skip_channels,
                inter_size=in_channels,
                nonlocal_mode='concatenation',
                sub_sample_factor=(1, 1),
            )

    def forward(self, x, skip=None, use_attention_layer=False):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            if use_attention_layer:
                x, _ = self.attention_layer(x, skip)
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x
    
    ###############################################################
    def get_attention_map(self, x, skip=None, use_attention_layer=False):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        a = None
        if skip is not None:
            if use_attention_layer:
                x, a = self.attention_layer(x, skip)
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return [x,a]
    ################################################################


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class AttentionUnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
            center=False,
    ):
        super().__init__()

        self.active_attention_layers = []
        
        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()
        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        
    def forward(self, *features):
        features = features[1:]    # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            if i+1 in self.active_attention_layers:
                skip = skips[i] if i < len(skips) else None
                x = decoder_block(x, skip, use_attention_layer=True)
            else:
                skip = skips[i] if i < len(skips) else None
                x = decoder_block(x, skip)
        return x
    
    #########################################################
    
    def get_attention_maps(self, features):
        attention_maps = []
        features = features[1:]    # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            if i+1 in self.active_attention_layers:
                skip = skips[i] if i < len(skips) else None
                y = decoder_block.get_attention_map(x, skip, use_attention_layer=True)
                x = y[0]
                a = y[1]
                if a is not None:
                    attention_maps.append(a)
            else:
                skip = skips[i] if i < len(skips) else None
                y = decoder_block.get_attention_map(x, skip)
                x = y[0]
                a = y[1]
                if a is not None:
                    attention_maps.append(a)                    

        return attention_maps
    ##############################################################


class AttentionUnet(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        n_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
        att_layers: Optional[list] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=n_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = AttentionUnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()
        
        if att_layers is not None:
            self.activate_attention_layers(att_layers)
        
        
    def get_attention_maps(self, x):
        x = self.encoder.forward(x)
        a = self.decoder.get_attention_maps(x)
        return a
    
    def activate_attention_layers(self, l):
        self.decoder.active_attention_layers = l
        
    def deactivate_attention_layers(self):
        self.decoder.active_attention_layers = []
        
    def active_attention_layers(self):
        return self.decoder.active_attention_layers
