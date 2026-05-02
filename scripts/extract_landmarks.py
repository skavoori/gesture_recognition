import cv2
import mediapipe as mp
import os
import pandas as pd
from tqdm import tqdm

def process_jester_data(dataset_path, output_csv_path, limit_folders=None):
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False, 
        max_num_hands=1,         
        min_detection_confidence=0.5
    )

    data_records = []

    # Get all video subdirectories (e.g., '1', '2', '3'...)
    all_folders = sorted([f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))])
    
    # Apply limit if specified (for Phase 1)
    video_folders = all_folders[:limit_folders] if limit_folders else all_folders
    
    phase_name = f"Test Phase ({limit_folders} folders)" if limit_folders else f"Full Phase ({len(video_folders)} folders)"
    print(f"\n--- Starting {phase_name} ---")

    # Iterate through each video folder with a progress bar
    for video_id in tqdm(video_folders):
        video_dir = os.path.join(dataset_path, video_id)
        # Grab all JPG frames in the folder and sort them numerically
        frames = sorted([f for f in os.listdir(video_dir) if f.endswith('.jpg')])

        for frame_name in frames:
            frame_path = os.path.join(video_dir, frame_name)
            image = cv2.imread(frame_path)

            if image is None:
                continue

            # OpenCV loads as BGR, MediaPipe requires RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            # If a hand is detected, extract the 21 landmarks
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Create a dictionary for this frame
                row_data = {'video_id': video_id, 'frame': frame_name}
                
                # Flatten the x, y, z coordinates into columns
                for i, landmark in enumerate(hand_landmarks.landmark):
                    row_data[f'x_{i}'] = landmark.x
                    row_data[f'y_{i}'] = landmark.y
                    row_data[f'z_{i}'] = landmark.z
                
                data_records.append(row_data)

    # Clean up MediaPipe resources
    hands.close()
    
    # Save to CSV and return the dataframe for validation
    if data_records:
        df = pd.DataFrame(data_records)
        df.to_csv(output_csv_path, index=False)
        print(f"\nData successfully saved to: {output_csv_path}")
        return df
    else:
        print("\nNo hands detected in this batch.")
        return None

if __name__ == "__main__":
    # --- UPDATE THIS PATH TO YOUR UNZIPPED FOLDER ---
    DATASET_PATH = "/Users/skavoori/projects/gesture_recognition/20bn-jester-v1" 
    
    TEST_CSV = "test_jester_coordinates.csv"
    FULL_CSV = "jester_hand_coordinates.csv"
    
    # ==========================================
    # PHASE 1: The Micro-Batch Test
    # ==========================================
    print("Initializing Phase 1: Micro-Batch Test (5 Folders)...")
    test_df = process_jester_data(DATASET_PATH, TEST_CSV, limit_folders=5)
    
    if test_df is not None:
        print("\nPhase 1 Successful! Here is a preview of your extracted coordinates:")
        # Print the first 3 rows and just the first few columns so it fits in the terminal
        print(test_df.head(3).iloc[:, :8]) 
        
        # ==========================================
        # PHASE 2: The Full Processing Run
        # ==========================================
        print("\n" + "="*60)
        print("ENVIRONMENT VERIFIED. READY FOR FULL DATASET PROCESSING.")
        print("="*60)
        
        user_input = input("Press ENTER to unleash Apple Silicon on all 148,000 clips, or type 'q' to quit: ")
        
        if user_input.lower() != 'q':
            print("\nInitializing Phase 2: Full Dataset Extraction...")
            process_jester_data(DATASET_PATH, FULL_CSV, limit_folders=None)
            print("\nCapstone preprocessing complete! You are ready to build the neural network.")
        else:
            print("\nAborting full run. You can inspect your test CSV locally.")
    else:
        print("\nPhase 1 failed to extract any landmarks. Please check your dataset path and ensure the frames are unzipped correctly.")