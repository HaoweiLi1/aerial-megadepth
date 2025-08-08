import numpy as np

# Load the npz file
data = np.load('megadepth_aerial_processed/0000/0000_000.jpeg.npz')

# View intrinsics
print("Camera intrinsics matrix:")
print(data['intrinsics'])

# View camera pose
print("Camera-to-world transformation:")
print(data['cam2world'])


import numpy as np

# Load your cam2world matrix (4x4 or 3x4)
cam2world = data['cam2world']

if cam2world.shape[-2:] == (4, 4):
    # Invert 4x4 homogeneous matrix
    world2cam = np.linalg.inv(cam2world)
    # Take only the first 3 rows for VGGT (3x4)
    extrinsic_for_vggt = world2cam[:3, :]
else:
    print("Camera-to-world transformation is not 4x4")
    exit()
# elif cam2world.shape[-2:] == (3, 4):
#     # For 3x4, need to construct 4x4 first
#     cam2world_4x4 = np.eye(4)
#     cam2world_4x4[:3, :] = cam2world
#     world2cam = np.linalg.inv(cam2world_4x4)
#     extrinsic_for_vggt = world2cam[:3, :]
print(extrinsic_for_vggt)

# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
# import os

# # Enable OpenEXR support in OpenCV
# os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# # Load the depth map
# depth_map = cv2.imread('megadepth_aerial_processed/0000/0000_000.jpeg.exr', cv2.IMREAD_UNCHANGED)

# # Visualize the depth map
# plt.figure(figsize=(10, 8))
# plt.imshow(depth_map, cmap='jet')
# plt.colorbar(label='Depth (meters)')
# plt.title('Depth Map Visualization')
# plt.show()

# # Print depth statistics
# print(f"Min depth: {np.min(depth_map):.2f} meters")
# print(f"Max depth: {np.max(depth_map):.2f} meters")
# print(f"Mean depth: {np.mean(depth_map):.2f} meters")

# # Load RGB image
# rgb_image = cv2.imread('megadepth_aerial_processed/0000/0000_000.jpeg.jpg')
# rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

# # Create side-by-side visualization
# fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# axes[0].imshow(rgb_image)
# axes[0].set_title('RGB Image')
# axes[0].axis('off')

# im = axes[1].imshow(depth_map, cmap='jet')
# axes[1].set_title('Depth Map')
# axes[1].axis('off')
# plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

# plt.tight_layout()
# plt.show()