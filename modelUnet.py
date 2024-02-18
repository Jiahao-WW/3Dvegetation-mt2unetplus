#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :modelUnet.py
@说明        :
@时间        :2023/03/12 
@作者        :Jiahao W
'''

from typing import Optional, Union, List
from encoders import get_encoder
from base import (
    MultitaskModel,
    RegressionHead,
    SegmentationHead,
    ClassificationHead,
)
from decoder import UnetPlusPlusDecoder


class UnetPlusPlus(MultitaskModel):

    def __init__(
        self,
        encoder_name: str = "resnext50_32x4d",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        if encoder_name.startswith("mit_b"):
            raise ValueError("UnetPlusPlus is not support encoder_name={}".format(encoder_name))

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetPlusPlusDecoder(
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

        self.regression_head = RegressionHead(
            in_channels=decoder_channels[-1],
            out_channels= 1,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = "unetplusplus-{}".format(encoder_name)
        self.initialize()
