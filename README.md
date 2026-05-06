# retinaface-inference

A self-contained, deployment-oriented RetinaFace face detection +
ArcFace alignment pipeline built on ONNX Runtime, plus an optional
**synthetic mask augmentation** stage for generating masked training
data for downstream face recognition.

This folder is a slimmed, library-shaped fork of the larger
`tiny-face-pytorch` workspace. It runs a pre-trained RetinaFace ONNX
model on real images and emits aligned 112×112 face crops suitable
for ArcFace / AdaFace / MagFace / any InsightFace-derivative
recognizer. The mask pipeline overlays surgical / N95 / cloth masks
onto faces (using dlib 68-point landmarks) before cropping, so you
can synthesize masked-face training sets without collecting them.

The detection math (preprocessing, PriorBox generation, box/landmark
decode, NMS, ArcFace 5-point similarity warp) is byte-equivalent to
the upstream `tiny-face-pytorch/onnx_inference.py`. The differences in
scope here are:

1. A clean library class ([retinaface.py](retinaface.py)) that takes a
   BGR `ndarray` in and returns detections + aligned crops in memory.
   No file I/O, no drawing, no JSON.
2. A thin CLI driver ([onnx_inference.py](onnx_inference.py)) that
   wraps the class with disk I/O, annotated-image saving, paired
   `cropped/` + `aligned/` output trees, and a `detections.json` log.
3. A masking + cropping pipeline
   ([generate_masked_images.py](generate_masked_images.py)) that
   composites synthetic masks onto faces (via dlib landmarks +
   masks under [masks/](masks/)) and then runs the same RetinaFace
   detect-and-align pass on the masked image.

---

## Repo layout

```
retinaface-inference/
├── README.md
├── requirements.txt
├── config.py                       # cfg_mnet / cfg_slim / cfg_rfb anchor configs
├── retinaface.py                   # RetinaFace library class (in-memory API)
├── onnx_inference.py               # CLI driver: detect + align only
├── generate_masked_images.py       # CLI driver: synthetic mask + detect + align
├── assets/
│   └── models/
│       ├── retinaface.onnx                          # FP32 RetinaFace (MobileNetV1-0.25)
│       └── shape_predictor_68_face_landmarks.dat    # dlib 68-pt landmark predictor
├── layers/
│   └── functions/
│       └── prior_box.py            # vectorized + cached PriorBox anchor generator
├── utils/
│   ├── align.py                    # ArcFace 5-point similarity warp (norm_crop)
│   ├── box_utils.py                # decode / decode_landmarks / cv2-backed nms
│   ├── general.py                  # draw_detections, get_output_path
│   ├── aux_functions.py            # mask_image: dlib landmarks -> mask overlay
│   ├── create_mask.py              # mask template warping helpers
│   ├── fit_ellipse.py              # ellipse fit used by the mask warper
│   └── read_cfg.py                 # masks/masks.cfg parser
└── masks/
    ├── masks.cfg                   # mask template anchor points
    ├── templates/                  # surgical / N95 / KN95 / cloth / gas PNGs (+left/right)
    └── textures/                   # optional cloth-mask textures (check / floral / fruits / others)
```

The `dlib` shape predictor and the RetinaFace ONNX both live under
[assets/models/](assets/models/) so the two CLI drivers can share
them without symlinks.

---

## Installation

A pinned `requirements.txt` is provided ([requirements.txt](requirements.txt)):

```
numpy==2.4.4
onnxruntime==1.24.1
onnxruntime_gpu==1.24.1
opencv_python==4.13.0.92
torch==2.6.0+cu124
tqdm==4.67.3
```

Install:

```bash
pip install -r requirements.txt
```


Skip dlib if you only plan to use [onnx_inference.py](onnx_inference.py).

`torch` is used only for the small bits of decode/scale math and for
the `PriorBox` cache tensor — there is no PyTorch model in the
inference path. The `+cu124` build is what bundles the CUDA 12.4 +
cuDNN 9 DLLs that [generate_masked_images.py](generate_masked_images.py)
adds to the DLL search path on import (see "GPU notes" below).

Tested on Python 3.11/3.12, Windows 11.

---

## Quick start — detection + alignment only

Run the bundled detect-only driver on a directory of images. Outputs
are written under `--output-path` (default `output/`):

```bash
python onnx_inference.py \
    --weights assets/models/retinaface.onnx \
    --network retinaface \
    --image-directory <input-dir> \
    --output-path output \
    --conf-threshold 0.4 \
    --nms-threshold 0.4 \
    --align-size 112 \
    --device cpu \
    -s
```

Note: the script's `--weights` default is still the legacy
`weights\retinaface.onnx` path. Pass
`--weights assets/models/retinaface.onnx` explicitly, or move/symlink
the ONNX into a `weights/` dir if you want to rely on the default.

Flags:

| Flag                 | Default                        | Meaning                                                                 |
| -------------------- | ------------------------------ | ----------------------------------------------------------------------- |
| `-w / --weights`     | `weights\retinaface.onnx`      | Path to the ONNX model file.                                            |
| `--network`          | `retinaface`                   | Architecture the ONNX file was exported from (`retinaface`/`slim`/`rfb`). |
| `--conf-threshold`   | `0.4`                          | Pre-NMS confidence cutoff. Use `0.02` to mirror WiderFace-AP convention.|
| `--nms-threshold`    | `0.4`                          | IoU threshold for NMS.                                                  |
| `--image-directory`  | *(required)*                   | Input directory; walked recursively for `.jpg`, `.jpeg`, `.png`.        |
| `--output-path`      | `output`                       | Output root.                                                            |
| `-s / --save-image`  | off                            | Also save annotated images and aggregate `detections.json`.             |
| `--align-size`       | `112`                          | Side length of the square aligned crop.                                 |
| `--device`           | `cpu`                          | `cpu` or `cuda` (CUDA needs `onnxruntime-gpu` + CUDA 12 + cuDNN 9).     |

Crops are saved unconditionally; only annotated images and
`detections.json` are gated by `-s`.

---

## Quick start — mask augmentation + detection + alignment

[generate_masked_images.py](generate_masked_images.py) merges a
dlib-based synthetic-mask overlay (surgical / N95 / cloth, with
random color, vertical offset, and rotation jitter) with the same
RetinaFace detect-and-align pass. dlib places the mask using 68-point
landmarks on the *un-masked* image; RetinaFace then runs on the
*post-mask* image so the bbox + 5 landmarks come from the same pixels
that get warped into the aligned crop.

```bash
python generate_masked_images.py \
    --input <input-dir-or-image> \
    --output output \
    --n_augmentations 3 \
    --shape_predictor assets/models/shape_predictor_68_face_landmarks.dat \
    --weights assets/models/retinaface.onnx \
    --conf-threshold 0.4 \
    --align-size 112 \
    --device cpu \
    --save_masked
```

For each input image, three randomized mask variations are generated;
each is then run through RetinaFace and yields paired raw + aligned
crops. With `--save_masked`, the full masked image is also saved
under `<output>/masked/` for inspection.

Flags:

| Flag                  | Default                                                          | Meaning                                                                                  |
| --------------------- | ---------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `--input`             | *(required)*                                                     | Input image **or** directory (walked recursively).                                       |
| `--output`            | `output`                                                         | Output root for `cropped/`, `aligned/`, `masked/`, `logs.csv`, `missed_images.txt`.      |
| `--n_augmentations`   | `1`                                                              | Mask variations per input image. Forced to 1 when `--no_mask` is set.                    |
| `--no_mask`           | off                                                              | Disable masking; run RetinaFace directly on the original image.                          |
| `--no_crop`           | off                                                              | Disable RetinaFace crop/align (mask-only mode for inspection).                           |
| `--save_masked`       | off                                                              | Also save the full masked image under `<output>/masked/`.                                |
| `--shape_predictor`   | `assets/models/shape_predictor_68_face_landmarks.dat`            | dlib 68-pt landmark predictor (skipped entirely if `--no_mask`).                         |
| `-w / --weights`      | `assets/models/retinaface.onnx`                                  | RetinaFace ONNX file.                                                                    |
| `--network`           | `retinaface`                                                     | `retinaface` / `slim` / `rfb`.                                                           |
| `--conf-threshold`    | `0.4`                                                            | RetinaFace pre-NMS confidence cutoff.                                                    |
| `--nms-threshold`     | `0.4`                                                            | IoU threshold for NMS.                                                                   |
| `--align-size`        | `112`                                                            | Side length of the aligned crop.                                                         |
| `--device`            | `cpu`                                                            | `cpu` or `cuda`.                                                                         |

For each augmentation `i`, mask params are sampled fresh:

- `mask_type` ∈ `{surgical, N95, cloth}`
- `color` from a 13-element palette (defined at the top of the script)
- `mask_y_offset` ∈ `[-10, 20)` pixels
- `mask_rotation_angle` ∈ `[-5, 5)` degrees

If dlib finds no face, or the mask compositing fails, that
augmentation is logged with `mask_success=False` and skipped. If
RetinaFace then produces no detection on the masked image, it is
logged as a "missed" image and added to `missed_images.txt`.

---

## As a Python library

```python
import cv2
from retinaface import RetinaFace

model = RetinaFace(
    model_path="assets/models/retinaface.onnx",
    network="retinaface",
    conf_threshold=0.4,
    nms_threshold=0.4,
    align_size=112,
    device="cpu",          # or "cuda"
)

image = cv2.imread("path/to/image.jpg")        # BGR uint8 HxWx3
faces = model.detect_and_align(image)

for f in faces:
    print(f["bbox"], f["score"], f["landmarks"].shape, f["aligned"].shape)
    # f["aligned"] is a 112x112 BGR uint8 ArcFace-aligned crop
```

If you don't need alignment, use `model.detect(image)` instead — it
returns an `(N, 15)` float32 array of `[x1, y1, x2, y2, score, lm0x,
lm0y, ..., lm4x, lm4y]` rows in original-image pixel coordinates,
sorted by score descending.

---

## Output layout

For an input tree like:

```
inputs/
├── personA/
│   ├── img001.png
│   └── img002.png
└── personB/
    └── img003.png
```

…[onnx_inference.py](onnx_inference.py) writes:

```
output/
├── cropped/                          # raw bbox crops (native size, clamped to image)
│   ├── personA/
│   │   ├── img001_face_000_conf0.987.jpg
│   │   └── img002_face_000_conf0.991.jpg
│   └── personB/
│       └── img003_face_000_conf0.964.jpg
├── aligned/                          # ArcFace-aligned 112x112 crops, paired 1:1 with cropped/
│   └── ... (mirrors cropped/ exactly)
├── <parent-dir-name>/                # annotated images, only with -s
│   └── ...
└── detections.json                   # only with -s
```

…and [generate_masked_images.py](generate_masked_images.py) writes:

```
output/
├── cropped/                          # crops from the (masked) image
│   └── personA/
│       ├── img001_aug0_face_000_conf0.913.jpg
│       ├── img001_aug1_face_000_conf0.928.jpg
│       └── img001_aug2_face_000_conf0.902.jpg
├── aligned/                          # paired 1:1 with cropped/
│   └── personA/
│       └── ...
├── masked/                           # full masked images (only with --save_masked)
│   └── personA/
│       ├── img001_aug0_masked.png
│       ├── img001_aug1_masked.png
│       └── img001_aug2_masked.png
├── logs.csv                          # per-augmentation log: mask_type / offset / rotation / success / error / output_path
└── missed_images.txt                 # appended list of inputs with no successful augmentation
```

The input folder structure is mirrored verbatim under `cropped/`,
`aligned/`, and `masked/`. The original image stem is folded into the
per-face filename (with an `_aug{i}` suffix from
[generate_masked_images.py](generate_masked_images.py)) so you do not
get an extra per-image directory level. Filenames pair exactly
one-to-one between `cropped/` and `aligned/`.

`detections.json` (saved by `onnx_inference.py -s`) is keyed by image
basename:

```json
{
    "img001.png": {
        "total_det": 2,
        "time_to_predict": 0.0612,
        "detections": [
            [x1, y1, x2, y2, score, lm0x, lm0y, lm1x, lm1y, lm2x, lm2y, lm3x, lm3y, lm4x, lm4y],
            ...
        ]
    },
    ...
}
```

`logs.csv` (written by `generate_masked_images.py`) has columns:
`image_path, mask_type, mask_y_offset, mask_rotation_angle, error,
mask_success, output_path, augmentation`. The CSV is opened in
append mode and flushed after each input image, so you can
interrupt and resume large runs without losing rows.

---

## Pipeline internals (RetinaFace)

The forward pass is identical to the upstream `onnx_inference.py`:

```
cv2.imread (BGR uint8)
  -> subtract mean [104, 117, 123]
  -> HWC -> CHW, add batch dim
  -> ort_session.run -> (loc, conf, landmarks)
  -> PriorBox(image_size).generate_anchors()        # cached per (H, W)
  -> decode boxes, decode_landmarks
  -> scale boxes by [W, H, W, H], landmarks by [W, H] * 5
  -> conf[:, 1] is the face score (softmax baked into eval-mode export)
  -> filter by --conf-threshold, top-k
  -> cv2.dnn.NMSBoxes
  -> top-k
  -> norm_crop (Umeyama similarity warp to ArcFace 5-pt template)
```

Key invariants — do **not** change without re-validating:

- **Preprocessing is BGR mean-subtract on uint8.** No `/255`, no RGB
  swap. The trained model expects exactly this.
- **Output tuple order is `(loc, conf, landmarks)`.** Hardcoded
  everywhere downstream.
- **`--network` must match the exported architecture.** A mismatch
  (`slim`/`rfb` config on a RetinaFace ONNX) decodes to garbage but
  inference completes silently — no error is raised.
- **FP16 inputs are auto-handled.** `_preprocess` checks the ONNX
  input type and casts to `float16` when needed.
- **Anchors are cached per `(H, W)`.** First call at a new resolution
  pays the generation cost (~2 ms at 1920×1080); subsequent calls at
  the same resolution are a dict lookup (~5 µs). The cached tensor
  lives on `self.device`, so CUDA mode pays the host→device transfer
  only once per resolution.
- **ArcFace alignment is bit-for-bit InsightFace-compatible.**
  [utils/align.py:norm_crop](utils/align.py) mirrors
  `insightface.utils.face_align.norm_crop` (same template, same
  scaling rules, same Umeyama math, same `cv2.warpAffine` call). The
  output is drop-in for any InsightFace recognizer.

---

## Pipeline internals (mask augmentation)

```
cv2.imread (BGR uint8)
  -> dlib.get_frontal_face_detector  -> face rect
  -> dlib.shape_predictor (68-pt)    -> landmarks
  -> utils/aux_functions.py:mask_image
       - pick template under masks/templates/<mask_type>{,_left,_right}.png
       - parse anchor points from masks/masks.cfg
       - utils/fit_ellipse.py + utils/create_mask.py warp template to face
       - apply random color tint, y-offset, rotation
       - alpha-composite onto image
  -> RetinaFace.detect_and_align    -> paired raw + aligned crops
```

The dlib detector and the shape predictor are loaded **once** per run
and shared across all augmentations. When `--no_mask` is passed,
neither is loaded — the ~95 MB shape predictor file is not even
opened — so detection-only runs do not pay the dlib startup cost.

---

## Performance notes

`RetinaFace.__init__` bakes in two production-hardening settings that
are **always on** (no flag — they are harmless on GPU and necessary on
CPU):

| Setting                                                    | Effect                                                                                       |
| ---------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `cv2.setNumThreads(1)`                                     | Stops OpenCV from contending with ORT's intra-op pool. Saves ~5 ms per image.                |
| `session.intra_op.allow_spinning = "0"`                    | Stops idle ORT threads from spinning and stealing CPU from the Python decode/PriorBox steps. |

[generate_masked_images.py](generate_masked_images.py) also calls
`cv2.setNumThreads(1)` at startup for the same reason.

---

## Threshold semantics

There are exactly two confidence-related knobs:

- **`--conf-threshold`** — applied **pre-NMS** on the raw model output.
  Anything below this is dropped before NMS sees it, so it never
  appears in `detections.json`, on the annotated image, or in
  `cropped/` / `aligned/`. Defaults:
  - `0.4` here (deployment / FR pipeline default).
  - `0.02` is the WiderFace AP-evaluation convention. Pass it
    explicitly if you need that for benchmarking.
- **`--nms-threshold`** — IoU threshold for NMS (default `0.4`).

---


## License & attribution

This folder is derived from
[`yakhyo/tiny-face-pytorch`](https://github.com/yakhyo/tiny-face-pytorch),
which is itself a PyTorch port of
[`biubug6/Pytorch_Retinaface`](https://github.com/biubug6/Pytorch_Retinaface).
ArcFace alignment math ([utils/align.py](utils/align.py)) is
bit-equivalent to
[`deepinsight/insightface`](https://github.com/deepinsight/insightface)'s
`face_align.norm_crop`, with the Umeyama similarity transform inlined
so `skimage` is not a runtime dependency. The mask augmentation
pipeline ([utils/aux_functions.py](utils/aux_functions.py),
[utils/create_mask.py](utils/create_mask.py),
[utils/fit_ellipse.py](utils/fit_ellipse.py))

