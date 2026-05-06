import cv2
import numpy as np
from typing import Tuple

import torch
from torch import nn, Tensor


def xywh2xyxy(boxes: Tensor | np.ndarray) -> Tensor | np.ndarray:
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right."""
    y = boxes.clone() if isinstance(boxes, torch.Tensor) else np.copy(boxes)
    y[..., 0] = boxes[..., 0] - boxes[..., 2] / 2  # top left x
    y[..., 1] = boxes[..., 1] - boxes[..., 3] / 2  # top left y
    y[..., 2] = boxes[..., 0] + boxes[..., 2] / 2  # bottom right x
    y[..., 3] = boxes[..., 1] + boxes[..., 3] / 2  # bottom right y

    return y


def xyxy2xywh(boxes: Tensor | np.ndarray) -> Tensor | np.ndarray:
    """Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right."""
    y = boxes.clone() if isinstance(boxes, torch.Tensor) else np.copy(boxes)
    y[..., 0] = (boxes[..., 0] + boxes[..., 2]) / 2  # x center
    y[..., 1] = (boxes[..., 1] + boxes[..., 3]) / 2  # y center
    y[..., 2] = boxes[..., 2] - boxes[..., 0]  # width
    y[..., 3] = boxes[..., 3] - boxes[..., 1]  # height

    return y


def _box_inter_union(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter

    return inter, union


def jaccard(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Return intersection-over-union (Jaccard index) between two sets of boxes.

    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        boxes1 (Tensor[N, 4]): first set of boxes
        boxes2 (Tensor[M, 4]): second set of boxes

    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """

    inter, union = _box_inter_union(boxes1, boxes2)
    iou = inter / union
    return iou


def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i)


def matrix_iof(a, b):
    """
    return iof of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    return area_i / np.maximum(area_a[:, np.newaxis], 1)


def match(
    overlap_threshold,
    gt_boxes,
    prior_boxes,
    variances,
    gt_labels,
    landmarks,
    loc_targets,
    conf_targets,
    landm_targets,
    batch_idx
):
    """
    Matches each prior box with the ground truth box of the highest jaccard overlap,
    encodes the bounding boxes, and updates the location and confidence targets.

    Args:
        overlap_threshold (float): The overlap threshold used when matching boxes.
        gt_boxes (tensor): Ground truth boxes, Shape: [num_objects, num_priors].
        prior_boxes (tensor): Prior boxes from priorbox layers, shape: [num_priors, 4].
        variances (tensor): Variances corresponding to each prior coord, shape: [num_priors, 4].
        gt_labels (tensor): Class labels for the image, shape: [num_objects].
        loc_targets (tensor): Tensor to be filled with encoded location targets.
        conf_targets (tensor): Tensor to be filled with matched indices for confidence predictions.
        batch_idx (int): Current batch index.
    """
    # Compute jaccard overlap between ground truth boxes and prior boxes
    overlaps = jaccard(gt_boxes, xywh2xyxy(prior_boxes))
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)

    # Ignore ground truths with low overlap
    valid_gt_idx = best_prior_overlap[:, 0] >= 0.2
    best_prior_idx_filter = best_prior_idx[valid_gt_idx, :]
    if best_prior_idx_filter.shape[0] <= 0:
        loc_targets[batch_idx] = 0
        conf_targets[batch_idx] = 0
        return

    # Find the best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_idx_filter.squeeze_(1)
    best_prior_overlap.squeeze_(1)

    # Ensure every ground truth matches with its prior of max overlap
    best_truth_overlap.index_fill_(0, best_prior_idx_filter, 2)
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j

    matches = gt_boxes[best_truth_idx]            # Shape: [num_priors,4]
    conf = gt_labels[best_truth_idx]               # Shape: [num_priors]
    conf[best_truth_overlap < overlap_threshold] = 0    # label as background
    loc = encode(matches, prior_boxes, variances)

    matches_landm = landmarks[best_truth_idx]
    landmarks = encode_landmarks(matches_landm, prior_boxes, variances)
    loc_targets[batch_idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_targets[batch_idx] = conf  # [num_priors] top class label for each prior

    landm_targets[batch_idx] = landmarks


def encode(matched, priors, variances):
    """
    Encode the coordinates of ground truth boxes based on jaccard overlap with the prior boxes.
    This encoded format is used during training to compare against the model's predictions.

    Args:
        matched (torch.Tensor): Ground truth coordinates for each prior in point-form, shape: [num_priors, 4].
        priors (torch.Tensor): Prior boxes in center-offset form, shape: [num_priors, 4].
        variances (list[float]): Variances of prior boxes

    Returns:
        torch.Tensor: Encoded boxes, Shape: [num_priors, 4]
    """

    # Calculate centers of ground truth boxes
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]

    # Normalize the centers with the size of the priors and variances
    g_cxcy /= (variances[0] * priors[:, 2:])

    # Calculate the sizes of the ground truth boxes
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]  # Use log to transform the scale

    # Concatenate normalized centers and sizes to get the encoded boxes
    encoded_boxes = torch.cat([g_cxcy, g_wh], dim=1)  # Concatenation along the last dimension

    return encoded_boxes


def encode_landmarks(matched, priors, variances):
    """
    Encode the variances from the prior boxes into the ground truth landmark coordinates.

    This function encodes the offset between the ground truth landmarks and the prior boxes (anchors) for use in
    localization loss during training. The encoding process adjusts the landmark coordinates using the variances
    and the dimensions of the prior boxes.

    Args:
        matched (tensor): Ground truth landmark coordinates matched to each prior box.
            Shape: [num_priors, 10], where each prior contains 5 landmark (x, y) pairs.
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors, 4], where each prior box contains (cx, cy, width, height).
        variances (list[float]): Variances used to scale the offset between the ground truth landmarks
            and the priors during encoding.

    Returns:
        g_cxcy (tensor): Encoded landmark offsets.
            Shape: [num_priors, 10], where each row contains the encoded (x, y) coordinates for 5 landmarks.
    """

    # Reshape matched landmarks into 5 points with 2 coordinates each (x, y)
    matched = matched.view(matched.size(0), 5, 2)

    # Extract priors' center coordinates (cx, cy) and width, height (w, h)
    priors_cx = priors[:, 0].view(-1, 1)
    priors_cy = priors[:, 1].view(-1, 1)
    priors_w = priors[:, 2].view(-1, 1)
    priors_h = priors[:, 3].view(-1, 1)

    # Compute the center offset between matched and prior landmarks
    g_cxcy = matched - torch.stack([priors_cx, priors_cy], dim=2)

    # Normalize by the variance-scaled width and height
    g_cxcy /= variances[0] * torch.stack([priors_w, priors_h], dim=2)

    # Flatten the landmark coordinates back to [num_priors, 10]
    g_cxcy = g_cxcy.view(g_cxcy.size(0), -1)

    return g_cxcy


def decode(loc, priors, variances):
    """
    Decode locations from predictions using priors to undo
    the encoding done for offset regression at train time.

    Args:
        loc (tensor): Location predictions for loc layers, shape: [num_priors, 4]
        priors (tensor): Prior boxes in center-offset form, shape: [num_priors, 4]
        variances (list[float]): Variances of prior boxes

    Returns:
        tensor: Decoded bounding box predictions
    """
    # Compute centers of predicted boxes
    cxcy = priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:]

    # Compute widths and heights of predicted boxes
    wh = priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])

    # Convert center, size to corner coordinates
    boxes = torch.empty_like(loc)
    boxes[:, :2] = cxcy - wh / 2  # xmin, ymin
    boxes[:, 2:] = cxcy + wh / 2  # xmax, ymax

    return boxes


def decode_landmarks(predictions, priors, variances):
    """
    Decode landmarks from predictions using prior boxes to reverse the encoding done during training.

    Args:
        predictions (tensor): Landmark predictions for localization layers.
            Shape: [num_priors, 10] where each prior contains 5 landmark (x, y) pairs.
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors, 4], where each prior has (cx, cy, width, height).
        variances (list[float]): Variances of the prior boxes to scale the decoded values.

    Returns:
        landmarks (tensor): Decoded landmark predictions.
            Shape: [num_priors, 10] where each row contains the decoded (x, y) pairs for 5 landmarks.
    """

    # Reshape predictions to [num_priors, 5, 2] to handle each pair (x, y) in a batch
    predictions = predictions.view(predictions.size(0), 5, 2)

    # Perform the same operation on all landmark pairs at once
    landmarks = priors[:, :2].unsqueeze(1) + predictions * variances[0] * priors[:, 2:].unsqueeze(1)

    # Flatten back to [num_priors, 10]
    landmarks = landmarks.view(landmarks.size(0), -1)

    return landmarks


def log_sum_exp(x):
    """
    Utility function for computing log_sum_exp.
    This function is used to compute the log of the sum of exponentials of input elements.

    Args:
        x (torch.Tensor): conf_preds from conf layers

    Returns:
        torch.Tensor: The result of the log_sum_exp computation.
    """
    return torch.logsumexp(x, dim=1, keepdim=True)


def nms(dets, threshold):
    """
    Apply Non-Maximum Suppression to reduce overlapping bounding boxes.

    Backed by cv2.dnn.NMSBoxes (C++ implementation, ~5-11x faster than the
    pure-numpy greedy loop). cv2 uses standard continuous-coordinate IoU
    rather than the legacy +1 pixel-inclusive convention, so at very low
    confidence thresholds (~0.02) a handful of borderline boxes near
    IoU=threshold may be kept that the legacy code suppressed. At deployment
    confidence thresholds (>= ~0.3) the kept-set is bit-identical.

    Args:
        dets (numpy.ndarray): Array of detections with each row as
            [x1, y1, x2, y2, score].
        threshold (float): IoU threshold for suppression.

    Returns:
        list[int]: Indices of bounding boxes retained after suppression,
            in score-descending order.
    """
    if dets.size == 0:
        return []

    # cv2.dnn.NMSBoxes expects (x, y, w, h) corner+size format.
    boxes_xywh = np.empty((dets.shape[0], 4), dtype=np.float32)
    boxes_xywh[:, 0] = dets[:, 0]
    boxes_xywh[:, 1] = dets[:, 1]
    boxes_xywh[:, 2] = dets[:, 2] - dets[:, 0]
    boxes_xywh[:, 3] = dets[:, 3] - dets[:, 1]

    # score_threshold=0 because the caller has already applied --conf-threshold.
    keep = cv2.dnn.NMSBoxes(
        boxes_xywh.tolist(),
        dets[:, 4].tolist(),
        score_threshold=0.0,
        nms_threshold=float(threshold),
    )
    if len(keep) == 0:
        return []
    return np.asarray(keep).flatten().tolist()
