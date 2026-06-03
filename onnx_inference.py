"""CLI driver that delegates all inference math to retinaface.RetinaFace.

Same on-disk outputs as onnx_inference.py (annotated images, paired
cropped/ + aligned/ face crops, detections.json) but with the ONNX
session, preprocessing, decode, NMS, and ArcFace alignment all
provided by the RetinaFace class — this script is just CLI + IO glue.
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
from tqdm import tqdm

from retinaface import RetinaFace
from utils.general import draw_detections, get_output_path


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="ONNX face detection CLI built on the RetinaFace library class."
    )
    parser.add_argument('-w', '--weights', type=str, default=r'assets/models/retinaface.onnx',
                        help='Path to the ONNX model weights.')
    parser.add_argument('--network', type=str, default='retinaface',
                        choices=['retinaface', 'slim', 'rfb'],
                        help='Architecture the ONNX file was exported from.')
    parser.add_argument('--conf-threshold', type=float, default=0.4,
                        help='Confidence threshold (deployment default 0.4; '
                             'pass 0.02 to mirror the WiderFace AP convention).')
    parser.add_argument('--nms-threshold', type=float, default=0.4,
                        help='IoU threshold for Non-Maximum Suppression.')
    parser.add_argument('--nist-1to1', action='store_true',
                        help='Configure for the NIST 1:1 recognition test: '
                             'forces --nms-threshold to 0.001 (overrides any '
                             'explicitly passed --nms-threshold).')
    parser.add_argument('--image-directory', type=str, required=True,
                        help='Directory of input images (walked recursively).')
    parser.add_argument('--output-path', type=str, default='output',
                        help='Output folder. Raw bbox crops go under '
                             '<output>/cropped/<image_stem>/ and ArcFace-aligned '
                             'crops under <output>/aligned/<image_stem>/.')
    parser.add_argument('-s', '--save-image', action='store_true',
                        help='Also save annotated images and detections.json '
                             'into <output>/. Crops are saved unconditionally.')
    parser.add_argument('--align-size', type=int, default=112,
                        help='Side length of the square aligned crop.')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Run on cpu (default) or cuda.')
    args = parser.parse_args()

    # NIST 1:1 recognition test needs almost all boxes kept through NMS so the
    # downstream recognizer can be evaluated on every candidate face.
    if args.nist_1to1:
        args.conf_threshold = 0.95
        args.nms_threshold = 0.001

    return args


def faces_to_array(faces):
    """Pack RetinaFace.detect_and_align() dicts back into the legacy (N, 15)
    array consumed by draw_detections and the JSON serializer."""
    if not faces:
        return np.zeros((0, 15), dtype=np.float32)
    rows = [
        np.concatenate([f["bbox"], [f["score"]], f["landmarks"].reshape(-1)])
        for f in faces
    ]
    return np.stack(rows).astype(np.float32)


def save_face_crops(image_path, input_root, image, faces, output_folder):
    """Write paired raw + aligned crops, mirroring the input folder structure
    exactly — no extra per-image folder, the original stem moves into the
    filename. For an input at `<input_root>/a/b/img.jpg`, crops land at:
        <output>/cropped/a/b/img_face_NNN_confX.XXX.jpg
        <output>/aligned/a/b/img_face_NNN_confX.XXX.jpg
    """
    if not faces:
        return 0

    rel_parent = os.path.relpath(os.path.dirname(image_path), input_root)
    if rel_parent == ".":
        rel_parent = ""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    cropped_dir = os.path.join(output_folder, "cropped", rel_parent)
    aligned_dir = os.path.join(output_folder, "aligned", rel_parent)
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


def process_image(model, image_path, args):
    image = cv2.imread(image_path)
    if image is None:
        print(f"  could not read {image_path}")
        return None, None

    start = time.time()
    faces = model.detect_and_align(image)
    time_taken = time.time() - start

    detections = faces_to_array(faces)
    #print(detections)

    save_face_crops(image_path, args.image_directory, image, faces, args.output_path)

    if args.save_image:
        draw_detections(image, detections)
        output_filename = get_output_path(image_path, args.output_path)
        cv2.imwrite(output_filename, image)

    return detections, time_taken


if __name__ == '__main__':
    args = parse_arguments()

    model = RetinaFace(
        model_path=args.weights,
        network=args.network,
        conf_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold,
        align_size=args.align_size,
        device=args.device,
    )

    if not os.path.isdir(args.image_directory):
        print(f"--image-directory must be a directory: {args.image_directory}")
        raise SystemExit(1)

    image_paths = []
    for dirs, _, files in os.walk(args.image_directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_paths.append(os.path.join(dirs, file))

    count = 0
    total_time = 0.0
    detection_details = {}

    for file_path in tqdm(image_paths, desc="Detecting faces", unit="img"):
        detections, time_taken = process_image(model, file_path, args)
        if time_taken is not None:
            detection_details[os.path.basename(file_path)] = {
                "detections": detections, "time_taken": time_taken,
            }
            count += 1
            total_time += time_taken

    if args.save_image and detection_details:
        json_ready = {
            k: {
                "total_det": len(v["detections"]),
                "time_to_predict": v["time_taken"],
                "detections": v["detections"].tolist(),
            }
            for k, v in detection_details.items()
        }
        os.makedirs(args.output_path, exist_ok=True)
        json_path = os.path.join(args.output_path, "detections.json")
        with open(json_path, 'w') as json_file:
            json.dump(json_ready, json_file, indent=4)

    if count > 0:
        print(f"Average inference time across {count} images: {total_time / count:.4f}s")
    else:
        print("No images processed.")
