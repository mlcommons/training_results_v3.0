# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2018-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.backbone import resnet
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.modeling.make_layers import group_norm
from maskrcnn_benchmark.modeling.make_layers import make_fc
from maskrcnn_benchmark.layers.nhwc import nchw_to_nhwc_transform, nhwc_to_nchw_transform
from maskrcnn_benchmark.utils.mlperf_logger import mllogger


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("ResNet50Conv5ROIFeatureExtractor")
class ResNet50Conv5ROIFeatureExtractor(nn.Module):
    def __init__(self, config):
        super(ResNet50Conv5ROIFeatureExtractor, self).__init__()

        resolution = config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = config.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = config.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        self.nhwc = config.NHWC
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
            is_nhwc = self.nhwc
        )

        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        head = resnet.ResNetHead(
            block_module=config.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=config.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=config.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=config.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=None,
            res2_out_channels=config.MODEL.RESNETS.RES2_OUT_CHANNELS,
            dilation=config.MODEL.RESNETS.RES5_DILATION,
            nhwc=self.nhwc
        )

        self.pooler = pooler
        self.head = head

    def forward(self, x, proposals):
##        if self.nhwc:
##            x = nchw_to_nhwc_transform(x)
        x = self.pooler(x, proposals)
        x = self.head(x)
        if self.nhwc:
            x = nhwc_to_nchw_transform(x)
        return x


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractor")
class FPN2MLPFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg):
        super(FPN2MLPFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
            is_nhwc = cfg.NHWC
        )
        self.nhwc = cfg.NHWC
        input_size = cfg.MODEL.BACKBONE.OUT_CHANNELS * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler
        self.fc6 = make_fc(input_size, representation_size, use_gn)
        if cfg.ENABLE_COMPLIANCE_LOG:
            mllogger.event(mllogger.constants.WEIGHTS_INITIALIZATION,
                metadata=dict(tensor='ROI_BOX_FEATURE_EXTRACTOR_fc6'))
        self.fc7 = make_fc(representation_size, representation_size, use_gn)
        if cfg.ENABLE_COMPLIANCE_LOG:
            mllogger.event(mllogger.constants.WEIGHTS_INITIALIZATION,
                metadata=dict(tensor='ROI_BOX_FEATURE_EXTRACTOR_fc7'))

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        ## TODO: we can get rid of this transpose by changing box head loss computation accordingly
        if self.nhwc:
            x = nhwc_to_nchw_transform(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPNXconv1fcFeatureExtractor")
class FPNXconv1fcFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg):
        super(FPNXconv1fcFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        self.pooler = pooler
        
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        in_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        conv_head_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_HEAD_DIM
        num_stacked_convs = cfg.MODEL.ROI_BOX_HEAD.NUM_STACKED_CONVS
        dilation = cfg.MODEL.ROI_BOX_HEAD.DILATION

        xconvs = []
        for ix in range(num_stacked_convs):
            xconvs.append(
                nn.Conv2d(
                    in_channels,
                    conv_head_dim,
                    kernel_size=3,
                    stride=1,
                    padding=dilation,
                    dilation=dilation,
                    bias=False if use_gn else True
                )
            )
            in_channels = conv_head_dim
            if use_gn:
                xconvs.append(group_norm(in_channels))
            xconvs.append(nn.ReLU(inplace=True))

        self.add_module("xconvs", nn.Sequential(*xconvs))
        for modules in [self.xconvs,]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if not use_gn:
                        torch.nn.init.constant_(l.bias, 0)

        input_size = conv_head_dim * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.fc6 = make_fc(input_size, representation_size, use_gn=False)

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.xconvs(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        return x


def make_roi_box_feature_extractor(cfg):
    func = registry.ROI_BOX_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg)
