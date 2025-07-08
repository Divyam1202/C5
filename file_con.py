import os
import rawpy
import imageio.v3 as iio
import json
import numpy as np
import tifffile

# Input/output directories
input_dir = r"C:\Users\Divyam Chandak\Desktop\demo"
output_dir = r"C:\Users\Divyam Chandak\Desktop\C5\demo_images"
os.makedirs(output_dir, exist_ok=True)


def extract_illuminant_metadata(tiff_path):
    """Extract illuminant metadata from a TIFF/DNG file as 3-channel RGB gains."""
    with tifffile.TiffFile(tiff_path) as tif:
        try:
            tags = tif.pages[0].tags
            wb_tag = tags.get('AsShotNeutral') or tags.get('WhiteLevel')
            if wb_tag:
                neutral = wb_tag.value
                if isinstance(neutral, (list, tuple, np.ndarray)):
                    # Take only first 3 values (R, G, B)
                    gains = [1.0 / max(n, 1e-6) for n in neutral[:3]]
                    return gains
        except Exception as e:
            print(f"Warning: Metadata read error in {os.path.basename(tiff_path)} — {e}")
    return [1.0, 1.0, 1.0]



def extract_sensor_name(tiff_path):
    """Attempt to extract sensor or camera name from EXIF metadata."""
    try:
        with tifffile.TiffFile(tiff_path) as tif:
            tags = tif.pages[0].tags
            make = tags.get('Make')
            model = tags.get('Model')
            sensor_name = ''

            if make and model:
                sensor_name = f"{make.value}_{model.value}"
            elif model:
                sensor_name = f"{model.value}"
            elif make:
                sensor_name = f"{make.value}"
            else:
                sensor_name = "unknown_sensor"

            # Clean up string
            sensor_name = sensor_name.strip().lower().replace(" ", "_")
            return sensor_name
    except Exception as e:
        print(f"Warning: Could not extract sensor name from {os.path.basename(tiff_path)} — {e}")
        return "unknown_sensor"


def generate_dummy_histogram(size=64):
    """Generate a dummy histogram shaped (H, W, 2) for RGB + edge."""
    hist_rgb = np.random.rand(size, size)
    hist_edges = np.random.rand(size, size)
    hist = np.stack([hist_rgb, hist_edges], axis=-1)
    return hist.astype(np.float32)



def process_raw_images():
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith((".dng", ".tiff", ".tif")):
            continue

        raw_path = os.path.join(input_dir, filename)
        base_name = os.path.splitext(filename)[0]

        # Extract actual sensor name
        sensor_name = extract_sensor_name(raw_path)

        # Final modified name
        modified_name = f"{base_name}_sensorname_{sensor_name}"

        output_img_path = os.path.join(output_dir, modified_name + ".png")
        output_json_path = os.path.join(output_dir, modified_name + "_metadata.json")
        output_hist_path = os.path.join(output_dir, modified_name + "_histogram.npy")

        # Convert RAW to PNG
        with rawpy.imread(raw_path) as raw:
            rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=8)
            iio.imwrite(output_img_path, rgb)

        # Extract WB metadata
        illuminant = extract_illuminant_metadata(raw_path)

        # Save metadata JSON
        metadata = {
            "illuminant_color_raw": illuminant,
            "sensor_name": sensor_name
        }
        with open(output_json_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save dummy histogram
        hist = generate_dummy_histogram()
        np.save(output_hist_path, hist)

        print(f"Processed: {filename} -> {modified_name}.(png/json/npy)")


# Run the function
process_raw_images()
