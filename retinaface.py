"""In-memory RetinaFace detection + ArcFace alignment.

A library-shaped wrapper around the same ONNX inference path used by
``onnx_inference.py``. Takes BGR ndarrays, returns detections (and
optionally aligned 112x112 crops) in memory. No CLI, no disk IO.

The math is identical to ``FaceONNXInference`` — same preprocessing,
PriorBox, decode, NMS, and ArcFace warp. The differences are scope:
this class drops file-path I/O, drawing, JSON, and timing prints, and
defaults ``conf_threshold`` to 0.4 (deployment) rather than 0.02
(WiderFace AP convention).
"""

import cv2
import numpy as np
import onnxruntime as ort
import torch

from config import get_config
from layers import PriorBox
from utils.align import norm_crop
from utils.box_utils import decode, decode_landmarks, nms


class RetinaFace:
    def __init__(
        self,
        model_path: str = "weights/retinaface.onnx",
        network: str = "retinaface",
        conf_threshold: float = 0.4,
        nms_threshold: float = 0.4,
        pre_nms_topk: int = 5000,
        post_nms_topk: int = 750,
        align_size: int = 112,
        device: str = "cpu",
    ) -> None:
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk
        self.align_size = align_size
        self.device = torch.device(device)

        cv2.setNumThreads(1)
        so = ort.SessionOptions()
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")

        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.ort_session = ort.InferenceSession(
            model_path, sess_options=so, providers=providers
        )
        self.input_name = self.ort_session.get_inputs()[0].name
        self.input_type = self.ort_session.get_inputs()[0].type

        self.cfg = get_config(network)
        self._prior_cache: dict = {}

    def _get_priors(self, image_size):
        priors = self._prior_cache.get(image_size)
        if priors is None:
            priors = PriorBox(self.cfg, image_size=image_size).generate_anchors()
            priors = priors.to(self.device)
            self._prior_cache[image_size] = priors
        return priors

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        rgb_mean = np.array([104, 117, 123], dtype=np.float32)
        x = image.astype(np.float32)
        x -= rgb_mean
        x = x.transpose(2, 0, 1)
        x = np.expand_dims(x, axis=0)
        if "float16" in self.input_type:
            x = x.astype(np.float16)
        return x

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Run detection on a BGR uint8 HxWx3 ndarray.

        Returns ``(N, 15)`` float32: ``[x1, y1, x2, y2, score,
        lm0x, lm0y, ..., lm4x, lm4y]`` in original-image pixel
        coordinates, sorted by score descending. Returns shape
        ``(0, 15)`` when no faces survive threshold + NMS.
        """
        h, w = image.shape[:2]
        x = self._preprocess(image)
        outputs = self.ort_session.run(None, {self.input_name: x})
        loc = outputs[0].squeeze(0)
        conf = outputs[1].squeeze(0)
        lms = outputs[2].squeeze(0)

        priors = self._get_priors((h, w))
        loc_t = torch.from_numpy(loc).to(self.device, non_blocking=True)
        lms_t = torch.from_numpy(lms).to(self.device, non_blocking=True)

        boxes = decode(loc_t, priors, self.cfg["variance"])
        landmarks = decode_landmarks(lms_t, priors, self.cfg["variance"])

        bbox_scale = torch.tensor([w, h] * 2, device=self.device)
        landmark_scale = torch.tensor([w, h] * 5, device=self.device)
        boxes = (boxes * bbox_scale).cpu().numpy()
        landmarks = (landmarks * landmark_scale).cpu().numpy()

        scores = conf[:, 1]

        keep_mask = scores > self.conf_threshold
        boxes = boxes[keep_mask]
        landmarks = landmarks[keep_mask]
        scores = scores[keep_mask]
        if scores.size == 0:
            return np.zeros((0, 15), dtype=np.float32)

        order = scores.argsort()[::-1][: self.pre_nms_topk]
        boxes = boxes[order]
        landmarks = landmarks[order]
        scores = scores[order]

        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(dets, self.nms_threshold)
        dets = dets[keep][: self.post_nms_topk]
        landmarks = landmarks[keep][: self.post_nms_topk]

        return np.concatenate((dets, landmarks), axis=1).astype(np.float32, copy=False)

    def detect_and_align(self, image: np.ndarray) -> list:
        """Run detection and return aligned 112x112 crops in memory.

        Returns a list of dicts (one per face, score-descending):
          - ``bbox``: ndarray(4,) float32, [x1, y1, x2, y2]
          - ``score``: float
          - ``landmarks``: ndarray(5, 2) float32
          - ``aligned``: ndarray(align_size, align_size, 3) uint8 BGR

        Faces whose alignment fails are dropped.
        """
        dets = self.detect(image)
        results = []
        for det in dets:
            landmarks = det[5:15].reshape(5, 2)
            try:
                aligned = norm_crop(image, landmarks, image_size=self.align_size)
            except Exception:
                continue
            results.append({
                "bbox": det[0:4].copy(),
                "score": float(det[4]),
                "landmarks": landmarks.copy(),
                "aligned": aligned,
            })
        return results
