import os
from src.aug_ops import set_sampling_params, map_raw_images

# --- 1. Set input/output paths ---
xyz_img_dir = r"C:\Users\Divyam Chandak\Desktop\C5\demo_images"
output_dir = r"C:\Users\Divyam Chandak\Desktop\C5\augmented_output"
os.makedirs(output_dir, exist_ok=True)

# --- 2. Choose target cameras ---
target_cameras = ["All"]

# --- 3. Set augmentation sampling parameters ---
params = set_sampling_params(im_per_scene_per_camera=1,
                    intensity_transfer=False,
                    target_aug_im_num=5000,
                    excluded_camera_models=None,
                    excluded_datasets=None,
                    save_as_16_bits=True,
                    remove_saturated_pixels=False,
                    saturation_level=0.97,
                    output_img_size=None,
                    cropping=True,
                    color_temp_balance=True,
                    lambda_r=0.7,
                    lambda_g=1.2,
                    k=15)


# --- 4. Run augmentation ---
map_raw_images(
    xyz_img_dir=xyz_img_dir,
    target_cameras=target_cameras,
    output_dir=output_dir,
    params=params
)

print("✅ Raw-to-Raw data augmentation completed.")
