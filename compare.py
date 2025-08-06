import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

def load_image(path):
    """Load image as numpy array."""
    img = Image.open(path)
    return np.array(img)

def normalize_depth(depth):
    """Normalize depth to 0-1 range."""
    if len(depth.shape) == 3:
        depth = depth.mean(axis=2)  # Convert to grayscale
    return (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

def compute_metrics(pred_depth, gt_depth):
    """Compute metrics between predicted and ground truth depth."""
    # Resize if needed
    if pred_depth.shape != gt_depth.shape:
        h, w = gt_depth.shape[:2]
        pred_depth = np.array(Image.fromarray(pred_depth).resize((w, h), Image.BILINEAR))
    
    # Normalize both
    pred_norm = normalize_depth(pred_depth)
    gt_norm = normalize_depth(gt_depth)
    
    # Compute metrics
    diff = np.abs(pred_norm - gt_norm)
    mae = np.mean(diff)
    rmse = np.sqrt(np.mean(diff ** 2))
    
    return mae, rmse

def compare_depths(rgb_path, vggt_depth_path, aerial_depth_path, output_path=None):
    """
    Compare VGGT depth with aerial depth (ground truth).
    
    Args:
        rgb_path: Path to RGB image (JPEG)
        vggt_depth_path: Path to VGGT depth map (PNG)
        aerial_depth_path: Path to aerial depth map (PNG) - used as ground truth
        output_path: Optional path to save visualization
    """
    # Load images
    rgb = load_image(rgb_path)
    vggt_depth = load_image(vggt_depth_path)
    aerial_depth = load_image(aerial_depth_path)
    
    # Compute metrics
    mae, rmse = compute_metrics(vggt_depth, aerial_depth)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # RGB image
    axes[0].imshow(rgb)
    axes[0].set_title('RGB Image', fontsize=14)
    axes[0].axis('off')
    
    # VGGT depth
    im1 = axes[1].imshow(vggt_depth, cmap='viridis')
    axes[1].set_title(f'VGGT Depth\nMAE: {mae:.3f}, RMSE: {rmse:.3f}', fontsize=14)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    # Aerial depth (GT)
    im2 = axes[2].imshow(aerial_depth, cmap='viridis')
    axes[2].set_title('Aerial Depth (GT)', fontsize=14)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    return mae, rmse

# Example usage
if __name__ == "__main__":
    # Single comparison
    compare_depths(
        rgb_path="0001_001.jpeg",
        vggt_depth_path="depth_vggt/0001_001.png",
        aerial_depth_path="0001_001_depth.png",
        output_path="depth_comparison.png"
    )
    
    # Batch processing
    def batch_compare(input_dir, output_dir):
        """Process multiple images in a directory."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Find all RGB images
        for rgb_file in input_path.glob("*.jpg"):
            stem = rgb_file.stem
            vggt_path = input_path / f"{stem}_vggt.png"
            aerial_path = input_path / f"{stem}_aerial.png"
            
            if vggt_path.exists() and aerial_path.exists():
                output_file = output_path / f"{stem}_comparison.png"
                mae, rmse = compare_depths(rgb_file, vggt_path, aerial_path, output_file)
                print(f"{stem}: MAE={mae:.4f}, RMSE={rmse:.4f}")
            else:
                print(f"Skipping {stem}: missing depth files")