# debug_npz.py
import numpy as np
import os.path as osp
import os
# Load and inspect the NPZ structure
data = np.load("data_splits/aerial_megadepth_val.npz", allow_pickle=True)

print("Keys in NPZ:", data.files)
print("\n--- Scenes Info ---")
scenes = data['scenes']
print(f"Scenes type: {type(scenes)}")
print(f"Scenes shape: {scenes.shape if hasattr(scenes, 'shape') else 'N/A'}")
print(f"Scenes dtype: {scenes.dtype if hasattr(scenes, 'dtype') else 'N/A'}")
print(f"First 10 scenes: {scenes[:10]}")

print("\n--- Pairs Info ---")
pairs = data['pairs']
print(f"Pairs type: {type(pairs)}")
print(f"Pairs shape: {pairs.shape if hasattr(pairs, 'shape') else 'N/A'}")
print(f"Pairs dtype: {pairs.dtype if hasattr(pairs, 'dtype') else 'N/A'}")

# Check if pairs is a structured array
if hasattr(pairs, 'dtype') and pairs.dtype.names:
    print(f"Pairs fields: {pairs.dtype.names}")
    if 'scene_id' in pairs.dtype.names:
        print(f"First 5 scene_ids: {pairs['scene_id'][:5]}")
else:
    # Maybe it's a dict or list?
    if isinstance(pairs, dict):
        print("Pairs is a dict with keys:", pairs.keys())
    elif isinstance(pairs, (list, tuple)) and len(pairs) > 0:
        print(f"First pair example: {pairs[0]}")

print("\n--- Check Your Actual Folders ---")
data_root = "/home/haowei/Documents/aerial-megadepth/megadepth_aerial_processed"
actual_folders = sorted([d for d in os.listdir(data_root) 
                         if osp.isdir(osp.join(data_root, d))])
print(f"Actual folders in {data_root}:")
print(actual_folders)

# Check if your folders match any scenes in NPZ
matching_scenes = [s for s in scenes if s in actual_folders]
print(f"\nMatching scenes: {matching_scenes}")