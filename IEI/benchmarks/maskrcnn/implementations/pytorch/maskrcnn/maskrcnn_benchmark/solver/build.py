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
import apex

from .lr_scheduler import WarmupMultiStepLR
from .fused_sgd import FusedSGD

def make_optimizer(cfg, model):
    params = []
    lr = cfg.SOLVER.BASE_LR
    weight_decay = cfg.SOLVER.WEIGHT_DECAY

    bias_params = []
    bias_lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
    bias_weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS

    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        if "bias" in key:
            bias_params.append(value)
        else:
            params.append(value)
    is_fp16 = (cfg.DTYPE == "float16")
    if is_fp16: # with FO16_Optimizer wrapper
        optimizer = FusedSGD(
            [
                {"params": params, "lr": lr, "weight_decay": weight_decay},
                {"params": bias_params, "lr": bias_lr, "weight_decay": bias_weight_decay}
            ],
            lr, momentum=cfg.SOLVER.MOMENTUM)
    else: # without FP16_Optimizer wrapper
        optimizer = apex.optimizers.FusedSGD(
            [
                {"params": params, "lr": lr, "weight_decay": weight_decay},
                {"params": bias_params, "lr": bias_lr, "weight_decay": bias_weight_decay}
            ],
            lr, momentum=cfg.SOLVER.MOMENTUM)

    return optimizer


def make_lr_scheduler(cfg, optimizer):
    return WarmupMultiStepLR(
        optimizer,
        cfg.SOLVER.STEPS,
        cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,
        scale_window=cfg.DYNAMIC_LOSS_SCALE_WINDOW
    )
