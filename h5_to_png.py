import h5py
import numpy as np
import cv2
import json

def h5_to_16bit_adaptive(h5_path, png_path, metadata_path=None):
    """
    Convert H5 to 16-bit PNG with adaptive range encoding.
    Maximizes precision by using the full 16-bit range for actual depth values.
    """
    with h5py.File(h5_path, 'r') as f:
        depth_map = np.array(f['depth'])
    
    # Get valid depth values
    valid_mask = np.isfinite(depth_map) & (depth_map > 0)
    valid_depths = depth_map[valid_mask]
    
    if len(valid_depths) == 0:
        # No valid depths
        cv2.imwrite(png_path, np.zeros_like(depth_map, dtype=np.uint16))
        return
    
    # Use 0.1% and 99.9% percentiles to exclude outliers
    min_depth = np.percentile(valid_depths, 0.1)
    max_depth = np.percentile(valid_depths, 99.9)
    
    # Normalize to 16-bit range (1-65535, reserve 0 for invalid)
    depth_normalized = (depth_map - min_depth) / (max_depth - min_depth)
    depth_normalized = np.clip(depth_normalized, 0, 1)
    depth_16bit = (depth_normalized * 65534 + 1).astype(np.uint16)
    
    # Set invalid pixels to 0
    depth_16bit[~valid_mask] = 0
    
    cv2.imwrite(png_path, depth_16bit)
    
    # Save metadata for accurate reconstruction
    if metadata_path:
        metadata = {
            'min_depth': float(min_depth),
            'max_depth': float(max_depth),
            'encoding': 'adaptive',
            'version': '1.0'
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    return depth_16bit, min_depth, max_depth

def png16_to_depth_adaptive(png_path, metadata_path):
    """
    Reconstruct original depth values from 16-bit PNG.
    """
    depth_16bit = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    min_depth = metadata['min_depth']
    max_depth = metadata['max_depth']
    
    # Reconstruct depth
    valid_mask = depth_16bit > 0
    depth_normalized = (depth_16bit.astype(np.float32) - 1) / 65534
    depth_map = depth_normalized * (max_depth - min_depth) + min_depth
    depth_map[~valid_mask] = 0
    
    return depth_map

# Example usage
# h5_path = "data/0001/sfm_output_localization/sfm_superpoint+superglue/localized_dense_metric/depths/0001_001.h5"
# png_path = "0001_001_depth.png"

# # Convert single file
# depth_16bit = h5_to_16bit_adaptive(h5_path, png_path)
# print(f"Converted {h5_path} to {png_path}")
# print(f"Depth range: {depth_16bit.min()} - {depth_16bit.max()} mm")

depth_16bit, min_d, max_d = h5_to_16bit_adaptive(
    "0001_001.h5", 
    "0001_001_depth.png", 
    "0001_001_depth_meta.json"
)

# # Reconstruct original
# depth_original = png16_to_depth_adaptive("depth.png", "depth_meta.json")