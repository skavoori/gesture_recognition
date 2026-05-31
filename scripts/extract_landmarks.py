import os

import cv2
import pandas as pd
from tqdm import tqdm

from hand_landmarker_utils import create_hand_landmarker, detect_hands_on_rgb
from mediapipe.tasks.python.vision.core import (
    vision_task_running_mode as running_mode_module,
)


def process_jester_data(dataset_path, output_csv_path, limit_folders=None):
    hand_landmarker = create_hand_landmarker(
        running_mode=running_mode_module.VisionTaskRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    data_records = []

    all_folders = sorted(
        f for f in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, f))
    )
    video_folders = all_folders[:limit_folders] if limit_folders else all_folders

    phase_name = (
        f"Test Phase ({limit_folders} folders)"
        if limit_folders
        else f"Full Phase ({len(video_folders)} folders)"
    )
    print(f"\n--- Starting {phase_name} ---")

    try:
        for video_id in tqdm(video_folders):
            video_dir = os.path.join(dataset_path, video_id)
            frames = sorted(
                f for f in os.listdir(video_dir) if f.endswith(".jpg")
            )
            frame_timestamp_ms = 0

            for frame_name in frames:
                frame_path = os.path.join(video_dir, frame_name)
                image = cv2.imread(frame_path)

                if image is None:
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = detect_hands_on_rgb(
                    image_rgb,
                    hand_landmarker,
                    timestamp_ms=frame_timestamp_ms,
                )
                frame_timestamp_ms += 33  # ~30 fps, monotonic per clip

                if results.hand_landmarks:
                    hand_landmarks = results.hand_landmarks[0]
                    row_data = {"video_id": video_id, "frame": frame_name}

                    for i, landmark in enumerate(hand_landmarks):
                        row_data[f"x_{i}"] = landmark.x
                        row_data[f"y_{i}"] = landmark.y
                        row_data[f"z_{i}"] = landmark.z

                    data_records.append(row_data)
    finally:
        hand_landmarker.close()

    if data_records:
        df = pd.DataFrame(data_records)
        df.to_csv(output_csv_path, index=False)
        print(f"\nData successfully saved to: {output_csv_path}")
        return df

    print("\nNo hands detected in this batch.")
    return None


if __name__ == "__main__":
    DATASET_PATH = "/Users/skavoori/projects/gesture_recognition/20bn-jester-v1"

    TEST_CSV = "test_jester_coordinates.csv"
    FULL_CSV = "jester_hand_coordinates.csv"

    print("Initializing Phase 1: Micro-Batch Test (5 Folders)...")
    test_df = process_jester_data(DATASET_PATH, TEST_CSV, limit_folders=5)

    if test_df is not None:
        print("\nPhase 1 Successful! Here is a preview of your extracted coordinates:")
        print(test_df.head(3).iloc[:, :8])

        print("\n" + "=" * 60)
        print("ENVIRONMENT VERIFIED. READY FOR FULL DATASET PROCESSING.")
        print("=" * 60)

        user_input = input(
            "Press ENTER to unleash Apple Silicon on all 148,000 clips, or type 'q' to quit: "
        )

        if user_input.lower() != "q":
            print("\nInitializing Phase 2: Full Dataset Extraction...")
            process_jester_data(DATASET_PATH, FULL_CSV, limit_folders=None)
            print("\nCapstone preprocessing complete! You are ready to build the neural network.")
        else:
            print("\nAborting full run. You can inspect your test CSV locally.")
    else:
        print(
            "\nPhase 1 failed to extract any landmarks. Please check your dataset path "
            "and ensure the frames are unzipped correctly."
        )
