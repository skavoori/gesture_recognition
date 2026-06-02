import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from paths import resolve_path

# This class is used to load the dataset into the memory and create a tensor of the correct shape
# It is used to train the model and validate the model
class JesterCoordinateDataset(Dataset):
    def __init__(self, coordinates_csv, labels_csv, class_list_csv, max_seq_length=30):
        coordinates_csv = resolve_path(coordinates_csv)
        labels_csv = resolve_path(labels_csv)
        class_list_csv = resolve_path(class_list_csv)

        print("Loading massive coordinate CSV into Memory...This could take a while...")
        self.df = pd.read_csv(coordinates_csv)
        
        print("Grouping frames by video sequence...")
        self.grouped_data = self.df.groupby('video_id')
        self.video_ids = list(self.grouped_data.groups.keys())
        
        self.max_seq_length = max_seq_length
        self.feature_cols = [col for col in self.df.columns if col.startswith(('x_', 'y_', 'z_'))]
        
        # --- NEW: Load the Real Labels ---
        print("Loading ground truth annotations...")
        classes_df = pd.read_csv(class_list_csv, header=None, names=['gesture_name'])
        self.class_to_idx = {name: idx for idx, name in enumerate(classes_df['gesture_name'])}
        
        labels_df = pd.read_csv(labels_csv, sep=';', header=None, names=['video_id', 'gesture_name'])
        labels_df['video_id'] = labels_df['video_id'].astype(str)
        self.video_to_label_str = dict(zip(labels_df['video_id'], labels_df['gesture_name']))
        
        # Filter our video_ids list to ONLY include videos we have labels for
        # This automatically drops the unlabelled Test set videos!
        self.video_ids = [vid for vid in self.video_ids if str(vid) in self.video_to_label_str]
        # ---------------------------------------------
        
    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        # The CSV saved the video_id as an integer, but the annotations use strings. 
        # We cast it here to ensure the dictionary lookup works.
        vid_id = str(self.video_ids[idx]) 
        
        # Extract features and pad/truncate
        video_frames = self.grouped_data.get_group(int(vid_id))[self.feature_cols].values
        video_frames = video_frames.astype(np.float32)
        
        # Create a tensor of the correct shape (max_seq_length, 63)
        seq_len = len(video_frames)
        if seq_len < self.max_seq_length:
            padding = np.zeros((self.max_seq_length - seq_len, 63), dtype=np.float32)
            video_tensor = np.vstack((video_frames, padding))
        else:
            video_tensor = video_frames[:self.max_seq_length, :]
            
        video_tensor = torch.tensor(video_tensor)
        
        # --- NEW: Map the numeric label ---
        gesture_string = self.video_to_label_str.get(vid_id, "No gesture")
        numeric_label = self.class_to_idx.get(gesture_string, 26) # 26 is usually "Doing other things"
        
        label_tensor = torch.tensor(numeric_label, dtype=torch.long) 
        
        return video_tensor, label_tensor

# This is the main function that is used to load the dataset into the memory and create a tensor of the correct shape
if __name__ == "__main__":
    COORD_CSV = "data/jester_hand_coordinates.csv"
    LABELS_CSV = "annotations/jester-v1-train.csv"
    CLASS_LIST_CSV = "annotations/jester-v1-labels.csv"
    
    print("Initializing PyTorch Dataset...")
    dataset = JesterCoordinateDataset(COORD_CSV, LABELS_CSV, CLASS_LIST_CSV, max_seq_length=30)
    
    print(f"\nTotal video sequences loaded: {len(dataset)}")
    
    # Grab the first sequence to verify
    sample_tensor, sample_label = dataset[0]
    print(f"Sample Tensor Shape: {sample_tensor.shape}")
    print(f"Sample Numeric Label: {sample_label.item()} (Maps to: {list(dataset.class_to_idx.keys())[sample_label.item()]})")