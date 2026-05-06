import cv2
import numpy as np
import shutil
from pathlib import Path


def draw_detections(original_image, detections, score_threshold=0.0):
    """
    Draws bounding boxes and landmarks on the image.

    Callers normally apply --conf-threshold pre-NMS so all surviving detections
    are draw-worthy; score_threshold defaults to 0 (no extra filter). Pass a
    higher value only when you want a stricter filter than --conf-threshold.

    Args:
        original_image (ndarray): The image on which to draw detections.
        detections (ndarray): Array of [x1,y1,x2,y2,score,*landmarks] rows.
        score_threshold (float): Optional extra confidence cutoff. Default 0.
    """

    # Colors for visualization
    LANDMARK_COLORS = [
        (0, 0, 255),    # Right eye (Red)
        (0, 255, 255),  # Left eye (Yellow)
        (255, 0, 255),  # Nose (Magenta)
        (0, 255, 0),    # Right mouth (Green)
        (255, 0, 0)     # Left mouth (Blue)
    ]
    BOX_COLOR = (0, 0, 255)
    TEXT_COLOR = (255, 255, 255)

    if score_threshold > 0:
        detections = detections[detections[:, 4] >= score_threshold]

    # print(f"#faces: {len(detections)}")

    # Slice arrays efficiently
    boxes = detections[:, 0:4].astype(np.int32)
    scores = detections[:, 4]
    landmarks = detections[:, 5:15].reshape(-1, 5, 2).astype(np.int32)

    for box, score, landmark in zip(boxes, scores, landmarks):
        # Draw bounding box
        cv2.rectangle(original_image, (box[0], box[1]), (box[2], box[3]), BOX_COLOR, 2)

        # Draw confidence score
        text = f"{score:.2f}"
        cx, cy = box[0], box[1] + 12
        cv2.putText(original_image, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, TEXT_COLOR)

        # Draw landmarks
        for point, color in zip(landmark, LANDMARK_COLORS):
            cv2.circle(original_image, point, 1, color, 4)


def get_output_path(input_path: str, output_dir: str) -> str:
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    
    if not input_path.exists():
        raise ValueError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        relative_parent = input_path.parent.name
        output_path = output_dir / relative_parent / input_path.name
    else:
        relative_path = input_path.relative_to(input_path)
        output_path = output_dir / relative_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    return str(output_path)
