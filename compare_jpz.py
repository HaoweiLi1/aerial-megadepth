import gzip
import json
import numpy as np
import os

def extract_jgz_info(jgz_path):
    """Extract camera parameters from VGGT's .jgz annotation file"""
    with gzip.open(jgz_path, 'r') as f:
        data = json.loads(f.read())
    
    print(f"\n=== JGZ File Info ({os.path.basename(jgz_path)}) ===")
    print(f"Number of sequences: {len(data)}")
    
    # Sample first sequence
    if data:
        seq_name = list(data.keys())[0]
        seq_data = data[seq_name]
        print(f"\nSample sequence '{seq_name}':")
        print(f"  Number of frames: {len(seq_data)}")
        
        # Sample first frame
        if seq_data:
            frame = seq_data[0]
            print(f"\n  Frame 0 contains keys: {list(frame.keys())}")
            
            if 'extri' in frame:
                extri = np.array(frame['extri'])
                print(f"    extrinsic shape: {extri.shape}")
                print(f"    extrinsic sample:\n{extri[:2, :]}")
            
            if 'intri' in frame:
                intri = np.array(frame['intri'])
                print(f"    intrinsic shape: {intri.shape}")
                print(f"    intrinsic sample:\n{intri}")
            
            if 'filepath' in frame:
                print(f"    filepath: {frame['filepath']}")
    
    return data

def extract_npz_info(npz_path):
    """Extract camera parameters from .npz file"""
    data = np.load(npz_path, allow_pickle=True)
    
    print(f"\n=== NPZ File Info ({os.path.basename(npz_path)}) ===")
    print(f"Keys in NPZ: {list(data.keys())}")
    
    for key in data.keys():
        arr = data[key]
        print(f"\n  '{key}':")
        print(f"    dtype: {arr.dtype}")
        print(f"    shape: {arr.shape}")
        
        # Show sample data for common camera parameter keys
        if key.lower() in ['extrinsic', 'extrinsics', 'extri', 'pose', 'camera_pose']:
            print(f"    sample extrinsic:\n{arr[0] if arr.ndim > 2 else arr}")
        elif key.lower() in ['intrinsic', 'intrinsics', 'intri', 'k', 'camera_matrix']:
            print(f"    sample intrinsic:\n{arr[0] if arr.ndim > 2 else arr}")
    
    return data

def compare_formats(jgz_data, npz_data):
    """Compare data organization between JGZ and NPZ formats"""
    print("\n=== Format Comparison ===")
    
    # Check if NPZ can be converted to VGGT's expected format
    print("\nVGGT expects from JGZ:")
    print("  - Dictionary of sequences")
    print("  - Each sequence: list of frame dictionaries")
    print("  - Each frame: {'extri': 3x4, 'intri': 3x3, 'filepath': str}")
    
    print("\nYour NPZ contains:")
    for key in npz_data.keys():
        print(f"  - {key}: {npz_data[key].shape}")
    
    print("\n=== Compatibility Check ===")
    
    # Check for essential camera parameters
    has_extrinsics = any(k.lower() in ['extrinsic', 'extrinsics', 'extri', 'pose'] 
                         for k in npz_data.keys())
    has_intrinsics = any(k.lower() in ['intrinsic', 'intrinsics', 'intri', 'k'] 
                         for k in npz_data.keys())
    
    print(f"✓ Has extrinsics: {has_extrinsics}")
    print(f"✓ Has intrinsics: {has_intrinsics}")
    
    # Check shapes
    for key in npz_data.keys():
        if 'extri' in key.lower():
            shape = npz_data[key].shape
            if len(shape) >= 2:
                last_dims = shape[-2:]
                if last_dims == (3, 4) or last_dims == (4, 4):
                    print(f"✓ Extrinsics shape compatible: {shape}")
                else:
                    print(f"✗ Extrinsics shape issue: {shape} (need 3x4 or 4x4)")
        
        if 'intri' in key.lower() or key.lower() == 'k':
            shape = npz_data[key].shape
            if len(shape) >= 2:
                last_dims = shape[-2:]
                if last_dims == (3, 3):
                    print(f"✓ Intrinsics shape compatible: {shape}")
                else:
                    print(f"✗ Intrinsics shape issue: {shape} (need 3x3)")

# Example usage
if __name__ == "__main__":
    # Paths to your files
    jgz_path = "apple_train.jgz"  # VGGT format
    npz_path = "megadepth_aerial_processed/0000/0000_000.jpeg.npz"  # Your format
    
    # # Extract data
    # try:
    #     jgz_data = extract_jgz_info(jgz_path)
    # except Exception as e:
    #     print(f"Could not load JGZ: {e}")
    #     jgz_data = None
    
    try:
        npz_data = extract_npz_info(npz_path)
    except Exception as e:
        print(f"Could not load NPZ: {e}")
        npz_data = None
    
    # # Compare formats
    # if jgz_data and npz_data:
    #     compare_formats(jgz_data, npz_data)