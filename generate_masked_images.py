"""Mask augmentation + RetinaFace crop/align pipeline.

Merges the VMScheduler-Augmentation mask pipeline (dlib 68-pt landmarks ->
synthetic surgical / N95 / cloth mask overlay via ``utils.aux_functions.mask_image``)
with the RetinaFace ONNX detector + ArcFace alignment already shipped in
this folder (see ``retinaface.py`` and ``onnx_inference3.py``).

For each input image we optionally apply a mask, then run RetinaFace on the
(possibly masked) image and emit paired raw bbox crops + ArcFace-aligned
112x112 crops. The mask landmark detector (dlib) and the face crop detector
(RetinaFace) are independent: dlib runs on the un-masked image to place the
mask, RetinaFace runs on the post-mask image so the bbox + 5 landmarks come
from the same pixels we are aligning.

All non-RetinaFace crop styles (buffalo / original_cropping / scale_shift)
from the original VMScheduler script have been removed.
"""

import argparse
import csv
import os
import sys
import time


def _enable_cuda_dll_search():
    """Make torch's bundled CUDA / cuDNN DLLs visible to onnxruntime-gpu.
    Idempotent and harmless on CPU runs and on non-Windows.
    """
    if not hasattr(os, "add_dll_directory"):
        return
    try:
        import torch 
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    except Exception:
        return
    if os.path.isdir(torch_lib):
        try:
            os.add_dll_directory(torch_lib)
        except (OSError, FileNotFoundError):
            pass


_enable_cuda_dll_search()


import cv2
import dlib
import numpy as np
from tqdm import tqdm

from retinaface import RetinaFace
from utils.aux_functions import mask_image


COLOR = [
    "#fc1c1a", "#177ABC", "#94B6D2", "#A5AB81", "#DD8047",
    "#6b425e", "#e26d5a", "#c92c48", "#6a506d", "#ffc900",
    "#ffffff", "#000000", "#49ff00",
]


def list_images(directory):
    paths = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                paths.append(os.path.join(root, f))
    return paths


def apply_mask(args_cfg, image_bgr):
    """Run the dlib-based mask overlay on a single image.

    Returns the masked ndarray, or ``None`` when dlib finds no face / the
    overlay fails.
    """
    if image_bgr is None:
        return None
    masked, mask, _, _ = mask_image(image_bgr, args_cfg)
    if mask is None or masked is None:
        return None
    if isinstance(masked, (list, tuple)):
        if not masked:
            return None
        return masked[0]
    return masked


def save_face_crops(image, faces, image_path, input_root, output_root, image_stem_suffix=""):
    """Mirror the input folder structure under ``output_root/cropped`` and
    ``output_root/aligned``. Filenames pair 1:1 between the two trees.
    """
    if not faces:
        return 0

    rel_parent = os.path.relpath(os.path.dirname(image_path), input_root)
    if rel_parent == ".":
        rel_parent = ""
    stem = os.path.splitext(os.path.basename(image_path))[0] + image_stem_suffix
    cropped_dir = os.path.join(output_root, "cropped", rel_parent)
    aligned_dir = os.path.join(output_root, "aligned", rel_parent)
    h, w = image.shape[:2]

    saved = 0
    for f in faces:
        bbox = f["bbox"]
        score = f["score"]
        x1 = max(0, int(round(float(bbox[0]))))
        y1 = max(0, int(round(float(bbox[1]))))
        x2 = min(w, int(round(float(bbox[2]))))
        y2 = min(h, int(round(float(bbox[3]))))
        if x2 <= x1 or y2 <= y1:
            continue
        bbox_crop = image[y1:y2, x1:x2]

        if saved == 0:
            os.makedirs(cropped_dir, exist_ok=True)
            os.makedirs(aligned_dir, exist_ok=True)

        out_name = f"{stem}_face_{saved:03d}_conf{score:.3f}.jpg"
        cv2.imwrite(os.path.join(cropped_dir, out_name), bbox_crop)
        cv2.imwrite(os.path.join(aligned_dir, out_name), f["aligned"])
        saved += 1
    return saved


def process_image(img_path, base_config, detector, predictor, model,
                  augment_idx=0, do_mask=True, do_crop=True, save_masked_full=True):
    config = base_config.copy()
    rel_path = os.path.relpath(img_path, start=base_config['input_root'])
    if rel_path.startswith('..'):
        raise ValueError(
            f"Image {img_path!r} is outside input_root "
            f"{base_config['input_root']!r}; cannot derive a safe output subdir."
        )
    sub_dir = os.path.split(rel_path)[0]
    masked_full_dir = os.path.join(config['output_dir'], "masked", sub_dir)

    log_entry = {
        'image_path': img_path, 'mask_type': 'NA', 'mask_success': 'NA',
        'error': '', 'augmentation': augment_idx, 'output_path': 'None',
        'mask_y_offset': 'NA', 'mask_rotation_angle': 'NA',
    }
    missed = False

    image = cv2.imread(img_path)
    if image is None:
        log_entry['error'] = 'cv2.imread returned None (file may be corrupted)'
        log_entry['mask_success'] = False
        return log_entry, True

    try:
        current_image = image
        aug_suffix = f"_aug{augment_idx}" if augment_idx >= 0 else ""

        if do_mask:
            config['mask_type'] = np.random.choice(["surgical", "N95", "cloth"])
            config['color'] = np.random.choice(COLOR)
            config['mask_y_offset'] = int(np.random.randint(-10, 20))
            config['mask_rotation_angle'] = int(np.random.randint(-5, 5))
            log_entry.update({
                'mask_type': config['mask_type'],
                'mask_y_offset': config['mask_y_offset'],
                'mask_rotation_angle': config['mask_rotation_angle'],
            })
            config['detector'] = detector
            config['predictor'] = predictor
            config['model'] = config.get('model', '')

            masked_result = apply_mask(config, image)
            if masked_result is None:
                log_entry['mask_success'] = False
                log_entry['error'] += "Mask application failed. "
                return log_entry, True

            current_image = masked_result
            log_entry['mask_success'] = True

            if save_masked_full:
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                os.makedirs(masked_full_dir, exist_ok=True)
                masked_path = os.path.join(
                    masked_full_dir, f"{base_name}{aug_suffix}_masked.png"
                )
                cv2.imwrite(masked_path, current_image)
                log_entry['output_path'] = masked_path
        else:
            log_entry['mask_type'] = "None (Disabled)"
            log_entry['mask_success'] = "Skipped"

        if do_crop:
            faces = model.detect_and_align(current_image)
            if not faces:
                log_entry['error'] += "RetinaFace produced no detection above thresholds. "
                missed = True
            else:
                saved = save_face_crops(
                    image=current_image,
                    faces=faces,
                    image_path=img_path,
                    input_root=base_config['input_root'],
                    output_root=base_config['output_dir'],
                    image_stem_suffix=aug_suffix,
                )
                if saved == 0:
                    log_entry['error'] += "All RetinaFace crops degenerate. "
                    missed = True
                elif log_entry['output_path'] == 'None':
                    log_entry['output_path'] = "Cropped output(s)"

    except Exception as e:
        log_entry['mask_success'] = False
        log_entry['error'] += f"Processing Error: {e}\n"
        missed = True

    return log_entry, missed


def run_pipeline(args):
    if args.n_augmentations < 1:
        print("Number of augmentations cannot be less than 1. Exiting.")
        sys.exit(1)
    start = time.time()

    base_config = {
        'verbose': False,
        'code': '',
        'pattern': 'random',
        'pattern_weight': 0.5,
        'color_weight': 0.5,
        'model': args.shape_predictor,
        'output_dir': args.output,
    }

    if os.path.isfile(args.input):
        image_paths = [args.input]
        base_config['input_root'] = os.path.dirname(args.input)
    elif os.path.isdir(args.input):
        image_paths = list_images(args.input)
        base_config['input_root'] = args.input
    else:
        print(f"Error: Input {args.input} not found.")
        return

    total_images = len(image_paths)
    actual_augmentations = 1 if args.no_mask else args.n_augmentations
    total_ops = total_images * actual_augmentations

    print("PIPELINE STARTED")
    print("*" * 60)
    print(f"Input: {args.input}")
    print(f"Images Found: {total_images}")
    print(f"Augmentations per Image: {args.n_augmentations}")
    print(f"Masking Enabled: {not args.no_mask}")
    print(f"Cropping Enabled: {not args.no_crop}")
    print(f"RetinaFace weights: {args.weights}")
    print(f"Total Operations: {total_ops}")
    print("*" * 60)
    os.makedirs(base_config['output_dir'], exist_ok=True)

    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    if args.no_mask:
        # Skip the ~95 MB shape_predictor load and the dlib detector init
        # entirely -- they're only used by the mask overlay path.
        detector = None
        predictor = None
    else:
        if not os.path.exists(args.shape_predictor):
            raise FileNotFoundError(
                f"dlib predictor model not found at: {args.shape_predictor}"
            )
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(args.shape_predictor)

    print("Initializing RetinaFace ONNX session ...")
    model = RetinaFace(
        model_path=args.weights,
        network=args.network,
        conf_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold,
        align_size=args.align_size,
        align_scale=args.align_scale,
        align_shift_y=args.align_shift_y,
        device=args.device,
    )
    if args.device == 'cuda':
        active = model.ort_session.get_providers()
        if 'CUDAExecutionProvider' not in active:
            raise RuntimeError(
                "Asked for --device cuda but ORT silently fell back to "
                f"providers={active}. Check that onnxruntime-gpu is installed "
                "and that CUDA 12 + cuDNN 9 DLLs are resolvable (see the "
                "earlier ORT log line about which DLL was missing)."
            )
        print(f"  ORT providers active: {active}")

    csv_path = os.path.join(base_config['output_dir'], "logs.csv")
    csv_fieldnames = [
        'image_path', 'mask_type', 'mask_y_offset', 'mask_rotation_angle',
        'error', 'mask_success', 'output_path', 'augmentation',
    ]
    file_exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
    csv_file_handle = open(csv_path, mode='a', newline='', encoding='utf-8')
    csv_writer = csv.DictWriter(csv_file_handle, fieldnames=csv_fieldnames)
    if not file_exists:
        csv_writer.writeheader()

    missed_img = []
    do_mask = not args.no_mask
    do_crop = not args.no_crop

    with tqdm(total=total_ops, desc="Processing") as pbar:
        for img_path in image_paths:
            for i in range(actual_augmentations):
                log_entry, missed = process_image(
                    img_path=img_path,
                    base_config=base_config,
                    detector=detector,
                    predictor=predictor,
                    model=model,
                    augment_idx=i,
                    do_mask=do_mask,
                    do_crop=do_crop,
                    save_masked_full=args.save_masked,
                )
                csv_writer.writerow(log_entry)
                if missed:
                    missed_img.append(f"{img_path} (Aug {i})")
                pbar.update(1)
            csv_file_handle.flush()

    csv_file_handle.close()

    if missed_img:
        missed_path = os.path.join(base_config['output_dir'], 'missed_images.txt')
        with open(missed_path, 'a', encoding='utf-8') as f:
            for m in missed_img:
                f.write(m + "\n")

    elapsed = time.time() - start
    print(f"Time taken: {elapsed:.2f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Mask Augmentation + RetinaFace crop/align pipeline."
    )
    parser.add_argument('--input', type=str, required=True,
                        help="Input image file or directory.")
    parser.add_argument('--output', type=str, default='output',
                        help="Output directory. Crops go under <output>/cropped, "
                             "aligned crops under <output>/aligned, and masked "
                             "full images under <output>/masked (when enabled).")
    parser.add_argument('--n_augmentations', type=int, default=1,
                        help="Number of mask variations per image. Forced to 1 "
                             "when --no_mask is set.")
    parser.add_argument('--no_mask', action='store_true',
                        help="Disable masking; run RetinaFace directly on the "
                             "original image.")
    parser.add_argument('--no_crop', action='store_true',
                        help="Disable RetinaFace cropping/alignment.")
    parser.add_argument('--save_masked', action='store_true',
                        help="Also save the full masked image alongside the crops.")
    parser.add_argument('--shape_predictor', type=str,
                        default=os.path.join('assets', 'models',
                                             'shape_predictor_68_face_landmarks.dat'),
                        help="Path to dlib's shape_predictor_68_face_landmarks.dat.")
    parser.add_argument('-w', '--weights', type=str,
                        default=os.path.join('assets', 'models', 'retinaface.onnx'),
                        help="Path to the RetinaFace ONNX weights.")
    parser.add_argument('--network', type=str, default='retinaface',
                        choices=['retinaface', 'slim', 'rfb'],
                        help="Architecture the ONNX file was exported from.")
    parser.add_argument('--conf-threshold', type=float, default=0.4,
                        help="RetinaFace confidence threshold.")
    parser.add_argument('--nms-threshold', type=float, default=0.4,
                        help="IoU threshold for Non-Maximum Suppression.")
    parser.add_argument('--align-size', type=int, default=112,
                        help="Side length of the square aligned crop.")
    parser.add_argument('--align-scale', type=float, default=1.0,
                        help="Zoom factor for the ArcFace destination template. "
                             "< 1.0 zooms out (more padding / face context); "
                             "> 1.0 zooms in (tighter crop). Default 1.0 = "
                             "standard InsightFace behaviour.")
    parser.add_argument('--align-shift-y', type=float, default=0.0,
                        help="Vertical shift in output pixels applied to the "
                             "ArcFace destination template. Positive = face moves "
                             "UP (more chin visible); negative = face moves DOWN "
                             "(more forehead visible). Default 0.0 = no shift.")
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help="Run RetinaFace on cpu (default) or cuda.")

    args = parser.parse_args()
    run_pipeline(args)
