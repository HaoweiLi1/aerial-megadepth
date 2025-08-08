# merge_and_filter_npz.py
import os
import os.path as osp
import numpy as np

def merge_and_filter_npz_files(
    npz_files,
    output_npz_path,
    data_root
):
    """
    Merge multiple NPZ files and filter to only available scenes.
    
    Args:
        npz_files: List of NPZ file paths
        output_npz_path: Output path for merged/filtered NPZ
        data_root: Root directory containing scene folders
    """
    # Get available scenes from disk
    actual_dirs = sorted([d for d in os.listdir(data_root) 
                          if osp.isdir(osp.join(data_root, d))])
    print(f"Actual directories found: {actual_dirs}\n")
    
    # Collect all data from multiple NPZ files
    all_scenes = []
    all_pairs = []
    all_images = None
    all_images_scene_name = None
    
    for npz_file in npz_files:
        print(f"Loading {npz_file}...")
        data = np.load(npz_file, allow_pickle=True)
        
        scenes = data['scenes']
        pairs = data['pairs']
        images = data['images']
        images_scene_name = data.get('images_scene_name', None)
        
        print(f"  - Found {len(scenes)} scenes, {len(pairs)} pairs")
        print(f"  - Scenes: {scenes[:5]}...")
        
        # Store scenes with their original indices
        for scene_idx, scene in enumerate(scenes):
            scene_str = str(scene)  # Convert numpy string to regular string
            if scene_str in actual_dirs:
                print(f"    ✓ Scene {scene_str} is available")
                all_scenes.append({
                    'name': scene_str,
                    'old_id': scene_idx,
                    'source_file': npz_file,
                    'pairs': []
                })
        
        # Store pairs with scene info
        for pair_idx, pair in enumerate(pairs):
            # Pairs format: [scene_id, im1_id, im2_id, score]
            scene_id = int(pair[0])
            scene_name = str(scenes[scene_id])
            
            if scene_name in actual_dirs:
                # Find the scene in our list
                for scene_info in all_scenes:
                    if scene_info['name'] == scene_name and scene_info['source_file'] == npz_file:
                        scene_info['pairs'].append({
                            'im1_id': pair[1],
                            'im2_id': pair[2],
                            'score': pair[3] if len(pair) > 3 else 1.0
                        })
                        break
        
        # Keep images and images_scene_name (should be same across files)
        if all_images is None:
            all_images = images
            all_images_scene_name = images_scene_name
    
    print(f"\n--- Merging Results ---")
    print(f"Total available scenes: {len(all_scenes)}")
    
    # Create new scene array and pairs array
    new_scenes = np.array([s['name'] for s in all_scenes], dtype='<U4')
    
    # Create new pairs array with updated scene IDs
    new_pairs_list = []
    for new_scene_id, scene_info in enumerate(all_scenes):
        for pair in scene_info['pairs']:
            new_pair = [
                new_scene_id,  # New scene ID
                pair['im1_id'],
                pair['im2_id'],
                pair['score']
            ]
            new_pairs_list.append(new_pair)
    
    # Convert to numpy array (preserving object dtype for compatibility)
    new_pairs = np.array(new_pairs_list, dtype=object)
    
    print(f"Total pairs after filtering: {len(new_pairs)}")
    
    # Statistics per scene
    print("\n--- Pairs per scene ---")
    for scene_info in all_scenes:
        print(f"  {scene_info['name']}: {len(scene_info['pairs'])} pairs")
    
    # Save merged and filtered data
    np.savez(
        output_npz_path,
        scenes=new_scenes,
        images=all_images,
        pairs=new_pairs,
        images_scene_name=all_images_scene_name
    )
    
    print(f"\n✅ Saved merged/filtered NPZ to {output_npz_path}")
    print(f"  - Scenes: {len(new_scenes)}")
    print(f"  - Pairs: {len(new_pairs)}")
    
    return new_scenes, len(new_pairs)

if __name__ == "__main__":
    data_root = "/home/haowei/Documents/aerial-megadepth/megadepth_aerial_processed"
    
    # Merge both part1 and part2
    scenes, n_pairs = merge_and_filter_npz_files(
        npz_files=[
            "data_splits/aerial_megadepth_train_part1.npz",
            "data_splits/aerial_megadepth_train_part2.npz"
        ],
        output_npz_path="aerial_megadepth_train_merged.npz",
        data_root=data_root
    )
    
    # Verify the output
    print("\n--- Verifying output ---")
    data = np.load("aerial_megadepth_train_merged.npz", allow_pickle=True)
    print(f"Scenes in merged file: {data['scenes']}")
    print(f"Number of pairs: {len(data['pairs'])}")
    if len(data['pairs']) > 0:
        print(f"First pair: {data['pairs'][0]}")
        print(f"Last pair: {data['pairs'][-1]}")