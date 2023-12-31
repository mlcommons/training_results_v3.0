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
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark import _C
import itertools


def old_generate_mask_targets(segmentation_masks, clamped_idxs, anchors, mask_size):
    # proven to work codepath
    with torch.cuda.nvtx.range("D2H1"):
        segmentation_masks = segmentation_masks[clamped_idxs]
    polygons_list=[]
    for poly_obj in segmentation_masks.polygons:
        polygons_per_instance=[]
        for poly in poly_obj.polygons:
            polygons_per_instance.append(poly)
        polygons_list.append(polygons_per_instance)
    dense_coordinate_vec=torch.cat(list(itertools.chain(*polygons_list))).double()
    if len(polygons_list) == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    if len(polygons_list)>0:
        result = _C.generate_mask_targets(dense_coordinate_vec, polygons_list, anchors, mask_size)
        return result


def syncfree_generate_mask_targets(segmentation_masks, clamped_idxs, anchors, mask_size):
    polygons_list=[]
    for poly_obj in segmentation_masks.polygons:
        polygons_per_instance=[]
        for poly in poly_obj.polygons:
            polygons_per_instance.append(poly)
        polygons_list.append(polygons_per_instance)
    #polygons_per_segmask = torch.tensor([len(poly_obj) for poly_obj in segmentation_masks.polygons], dtype=torch.int32, pin_memory=True).to(device='cuda', non_blocking=True)
    #samples_per_segmask = torch.tensor([sum([p.numel() for p in poly_obj]) for poly_obj in segmentation_masks.polygons], dtype=torch.int32, pin_memory=True).to(device='cuda', non_blocking=True)
    if len(polygons_list) == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    if len(polygons_list)>0:
        result = _C.syncfree_generate_mask_targets(clamped_idxs, polygons_list, anchors, mask_size)
        return result


def project_masks_on_boxes(segmentation_masks, clamped_idxs, proposals, discretization_size, syncfree):
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.

    Arguments:
        segmentation_masks: an instance of SegmentationMask
        proposals: an instance of BoxList
    """
    masks = []
    M = discretization_size
    device = proposals.bbox.device
    proposals = proposals.convert("xyxy")
    #assert segmentation_masks.size == proposals.size, "{}, {}".format(
    #    segmentation_masks, proposals
    #)
    # CUDA implementation of RLE encoding needs to be fixed to support larger M
    if proposals.bbox.is_cuda and M < 32:
        if syncfree:
            masks = syncfree_generate_mask_targets(segmentation_masks, clamped_idxs, proposals.bbox, M)
        else:
            masks = generate_mask_targets(segmentation_masks, clamped_idxs, proposals.bbox, M)
        return masks
    else:
        with torch.cuda.nvtx.range("D2H1"):
            segmentation_masks = segmentation_masks[clamped_idxs]
        proposals = proposals.bbox.to(torch.device("cpu"))
        for segmentation_mask, proposal in zip(segmentation_masks, proposals):
            # crop the masks, resize them to the desired resolution and
            # then convert them to the tensor representation,
            # instead of the list representation that was used
            cropped_mask = segmentation_mask.crop(proposal)
            scaled_mask = cropped_mask.resize((M, M))
            mask = scaled_mask.convert(mode="mask")
            masks.append(mask)
        if len(masks) == 0:
            return torch.empty(0, dtype=torch.float32, device=device)
        return torch.stack(masks, dim=0).to(device, dtype=torch.float32)

class MaskRCNNLossComputation(object):
    def __init__(self, proposal_matcher, discretization_size, syncfree):
        """
        Arguments:
            proposal_matcher (Matcher)
            discretization_size (int)
        """
        self.proposal_matcher = proposal_matcher
        self.discretization_size = discretization_size
        self.syncfree = syncfree

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Mask RCNN needs "labels" and "masks "fields for creating the targets
        # target = target.copy_with_fields(["labels", "masks"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        #matched_targets = target[matched_idxs.clamp(min=0)]
        #matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_idxs

    def prepare_targets(self, proposals, targets, syncfree):
        labels = []
        for proposals_per_image in proposals:
            # Grab matched labels directly
            # labels_per_image = matched_targets.get_field("labels")
            labels_per_image = proposals_per_image.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            labels.append(labels_per_image)
        labels_pos = cat(labels, dim=0)

        masks, streams = [], []
        cs = torch.cuda.current_stream()
        multi_stream = True if len(targets) > 1 else False
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            if multi_stream:
                s = torch.cuda.Stream()
                s.wait_stream(cs)
                streams.append(s)
            else:
                s = cs
            with torch.cuda.stream(s):
                matched_idxs = proposals_per_image.get_field("matched_idxs")
                clamped_idxs = matched_idxs.clamp(min=0)
                # generated this directly
                # matched_idxs = matched_targets.get_field("matched_idxs")

                # this can probably be removed, but is left here for clarity
                # and completeness
                # neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
                # labels_per_image[neg_inds] = 0

                # mask scores are only computed on positive samples
                # positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)

                # Copy masks once instead of twice
                # NOTE: matched_idxs a mask into original array
                #       positive_inds a mask into reduced array of size matched_idxs
                # samples for mask head are all positive, no need to index with positive_inds
                segmentation_masks = targets_per_image.get_field("masks")
                # segmentation_masks = segmentation_masks[positive_inds]

                # positive_proposals = proposals_per_image[positive_inds]

                masks_per_image = project_masks_on_boxes(
                    segmentation_masks, clamped_idxs, proposals_per_image, self.discretization_size,
                    syncfree
                )

            masks.append(masks_per_image)
        for s in streams:
            cs.wait_stream(s)
        mask_targets = cat(masks, dim=0)

        return labels_pos, mask_targets

    def __call__(self, proposals, mask_logits, targets, weights, scale):
        """
        Arguments:
            proposals (list[BoxList])
            mask_logits (Tensor)
            targets (list[BoxList])

        Return:
            mask_loss (Tensor): scalar tensor containing the loss
        """
        mask_weights = None if weights is None else (cat(weights, dim=0) * scale).unsqueeze(1).unsqueeze(2)

        N, C, H, W = mask_logits.shape
	# negative sampls are already filtered out, so all samples are positive
        positive_inds = torch.arange(0, N, device = mask_logits.device)
        index_select_indices = positive_inds * mask_logits.size(1)

        _, _, _, targets = targets
        labels_pos, mask_targets = self.prepare_targets(proposals, targets, self.syncfree)

        # mask samples are all positive, no need to search for positive samples 
        #positive_inds = torch.nonzero(labels > 0).squeeze(1)
        #labels_pos = labels[positive_inds]

        # torch.mean (in binary_cross_entropy_with_logits) doesn't
        # accept empty tensors, so handle it separately
        if mask_targets.numel() == 0:
            return mask_logits.sum() * 0

        index_select_indices = (index_select_indices + labels_pos).view(-1)
        mask_logits_sampled = mask_logits.view(-1, H, W).index_select(0, index_select_indices).view(N, H, W)

        mask_loss = F.binary_cross_entropy_with_logits(
            mask_logits_sampled, mask_targets, weight=mask_weights
        )
        return mask_loss


def make_roi_mask_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    loss_evaluator = MaskRCNNLossComputation(
        matcher, cfg.MODEL.ROI_MASK_HEAD.RESOLUTION,
        cfg.SYNCFREE_ROI
    )

    return loss_evaluator
