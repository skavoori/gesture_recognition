import argparse
import sys
import urllib.request
from collections import deque

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    drawing_styles,
    drawing_utils,
)
from mediapipe.tasks.python.vision.core import image as mp_image_module
from mediapipe.tasks.python.vision.core import (
    vision_task_running_mode as running_mode_module,
)

from paths import ROOT
from model import GestureLSTM, GestureGRU

_HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
_HAND_LANDMARKER_PATH = ROOT / "models" / "hand_landmarker.task"


def _ensure_hand_landmarker_model() -> str:
    if _HAND_LANDMARKER_PATH.is_file():
        return str(_HAND_LANDMARKER_PATH)

    _HAND_LANDMARKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Downloading hand landmarker model to {_HAND_LANDMARKER_PATH}")
    urllib.request.urlretrieve(_HAND_LANDMARKER_URL, _HAND_LANDMARKER_PATH)
    return str(_HAND_LANDMARKER_PATH)


def _create_hand_landmarker() -> HandLandmarker:
    options = HandLandmarkerOptions(
        base_options=base_options_module.BaseOptions(
            model_asset_path=_ensure_hand_landmarker_model()
        ),
        running_mode=running_mode_module.VisionTaskRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    return HandLandmarker.create_from_options(options)

def run_live_inference(model_path, model_type, hidden_size):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Initializing Live Inference on: {device}")

    # 1. Load the Label Dictionary
    labels_csv = ROOT / "annotations" / "jester-v1-labels.csv"
    try:
        classes_df = pd.read_csv(labels_csv, header=None, names=['gesture_name'])
        class_names = classes_df['gesture_name'].tolist()
    except FileNotFoundError:
        print(f"[ERROR] Could not find labels file at {labels_csv}")
        sys.exit(1)

    # 2. Initialize the Model
    if model_type.upper() == 'GRU':
        model = GestureGRU(input_size=63, hidden_size=hidden_size, num_layers=2, num_classes=27).to(device)
    else:
        model = GestureLSTM(input_size=63, hidden_size=hidden_size, num_layers=2, num_classes=27).to(device)
    
    # Load the trained weights
    full_model_path = ROOT / "models" / model_path
    try:
        model.load_state_dict(torch.load(full_model_path, map_location=device))
        model.eval() # Freeze dropout layers for inference
        print(f"Successfully loaded {model_type} weights from {full_model_path}")
    except FileNotFoundError:
        print(f"[ERROR] Could not find model weights at {full_model_path}")
        sys.exit(1)

    # 3. Initialize MediaPipe Hand Landmarker (Tasks API; legacy mp.solutions removed in 0.10+)
    hand_landmarker = _create_hand_landmarker()
    hand_connections = HandLandmarksConnections.HAND_CONNECTIONS
    hand_landmark_style = drawing_styles.get_default_hand_landmarks_style()
    hand_connection_style = drawing_styles.get_default_hand_connections_style()

    # 4. Setup Rolling Buffer and OpenCV
    sequence = deque(maxlen=30)
    cap = cv2.VideoCapture(0)  # 0 is usually the built-in webcam
    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Grant camera access in System Settings > Privacy & Security > Camera.")
        hand_landmarker.close()
        sys.exit(1)

    frame_timestamp_ms = 0
    current_gesture = "Waiting for hand..."
    confidence = 0.0

    print("\n[INFO] Webcam activated. Press 'q' in the video window to quit.")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Flip horizontally for a natural selfie-view display
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp_image_module.Image(
                image_format=mp_image_module.ImageFormat.SRGB,
                data=np.ascontiguousarray(frame_rgb),
            )
            results = hand_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            frame_timestamp_ms += int(1000 / 30)

            if results.hand_landmarks:
                hand_landmarks = results.hand_landmarks[0]

                drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks,
                    hand_connections,
                    landmark_drawing_spec=hand_landmark_style,
                    connection_drawing_spec=hand_connection_style,
                )

                coords = []
                for lm in hand_landmarks:
                    coords.extend([lm.x, lm.y, lm.z])
            
                sequence.append(coords)

                # 5. Run Inference if we have a full 30-frame window
                if len(sequence) == 30:
                    input_tensor = torch.tensor(
                        np.array(sequence), dtype=torch.float32
                    ).unsqueeze(0).to(device)

                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probabilities = F.softmax(outputs, dim=1)
                        max_prob, predicted = torch.max(probabilities.data, 1)

                        confidence = max_prob.item() * 100

                        if confidence > 65.0:
                            current_gesture = class_names[predicted.item()]
            else:
                if len(sequence) > 0:
                    current_gesture = "Hand lost. Resetting..."
                sequence.clear()

            # 6. UI Overlay
            cv2.rectangle(frame, (0, 0), (640, 50), (0, 0, 0), -1)
            cv2.putText(
                frame,
                f"Gesture: {current_gesture}",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Conf: {confidence:.1f}%",
                (450, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("AppleTV Gesture Interface Prototype", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        hand_landmarker.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Hand Gesture Inference")
    # You will pass the specific model file generated by the sweep
    parser.add_argument('--model_path', type=str, required=True, help="Filename of the .pth model in the /models directory (e.g., best_lstm_h128_lr0.001.pth)")
    parser.add_argument('--model_type', type=str, default='LSTM', choices=['LSTM', 'GRU'])
    parser.add_argument('--hidden_size', type=int, default=128)
    
    args = parser.parse_args()
    run_live_inference(args.model_path, args.model_type, args.hidden_size)
