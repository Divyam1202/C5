import os
import torch
import logging
import argparse
import numpy as np
from torchvision.utils import save_image, make_grid
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from scipy.io import savemat

from src import dataset, c5, ops
from PIL import Image, ImageEnhance  # ✅ ADD THIS
from PIL import Image


def run_c5_inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cpu':
        torch.cuda.set_device(args.gpu)

    logging.info(f'Using device: {device}')

    # Load model
    net = c5.network(input_size=args.input_size,
                     learn_g=args.g_multiplier,
                     data_num=args.data_num,
                     device=str(device))
    model_path = os.path.join('models', args.model_name + '.pth')
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device=device)
    net.eval()
    logging.info(f'Model loaded: {model_path}')

    # Load dataset
    test_files = dataset.Data.load_files(args.in_tedir)
    test_dataset = dataset.Data(test_files, input_size=args.input_size,
                                 mode='testing', data_num=args.data_num,
                                 load_hist=args.load_hist)
    test_loader = DataLoader(test_dataset, batch_size=args.batchsize,
                             shuffle=False, num_workers=0)

    os.makedirs('results/' + args.model_name, exist_ok=True)

    result_pred = []
    result_gt = []
    filenames = []

    with torch.no_grad():
        for batch in test_loader:
            image = batch['image_rgb'].to(device)
            gt_ill = batch['gt_ill'].to(device)
            histogram = batch['histogram'].to(device)
            model_hist = batch['model_input_histograms'].to(device)
            file_names = batch['file_name']

            pred_ill, *_ = net(histogram, model_in_N=model_hist)

            for idx in range(image.size(0)):
                img_tensor = image[idx]
                correction = 1.0 / (pred_ill[idx] + 1e-6)
                corrected = img_tensor * correction.view(3, 1, 1)
                corrected = torch.clamp(corrected / corrected.max(), 0, 1)

                save_path = os.path.join('results', args.model_name,
                                         file_names[idx].replace('.png', '_corrected.png'))
                save_image(corrected, save_path)
                logging.info(f"Saved: {save_path}")

                result_pred.append(pred_ill[idx].cpu().numpy())
                result_gt.append(gt_ill[idx].cpu().numpy())
                filenames.append(file_names[idx])

    # Save .mat results
    save_dir = os.path.join('results', args.model_name)
    savemat(os.path.join(save_dir, 'gt.mat'), {'gt': np.array(result_gt)})
    savemat(os.path.join(save_dir, 'results.mat'), {'predicted': np.array(result_pred)})
    savemat(os.path.join(save_dir, 'filenames.mat'), {'filenames': filenames})
    logging.info("Saved all results")


def get_args():
    parser = argparse.ArgumentParser(description='Test C5 Model')
    parser.add_argument('--in_tedir', type=str, default='./testing_set/', help='Input test dir')
    parser.add_argument('--model_name', type=str, default='c5_model')
    parser.add_argument('--input_size', type=int, default=64)
    parser.add_argument('--data_num', type=int, default=7)
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--g_multiplier', type=bool, default=False)
    parser.add_argument('--load_hist', type=bool, default=True)
    return parser.parse_args()

# src/aug_ops.py -- NEW ADDITION (sampling_parmams and map_raw_images)

def set_sampling_params(**kwargs):
    """Return a dictionary of sampling parameters."""
    print("✅ Called set_sampling_params with:")
    for k, v in kwargs.items():
        print(f"  {k}: {v}")
    return kwargs


def map_raw_images(xyz_img_dir, target_cameras, output_dir, params):
    print("✅ Called map_raw_images with:")
    print(f"  xyz_img_dir: {xyz_img_dir}")
    print(f"  target_cameras: {target_cameras}")
    print(f"  output_dir: {output_dir}")
    print(f"  params: {params}")

    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(xyz_img_dir):
        if not fname.lower().endswith(".png"):
            continue

        fpath = os.path.join(xyz_img_dir, fname)
        try:
            img = Image.open(fpath)
            image_array = np.array(img)

            if image_array.ndim != 3 or image_array.shape[2] != 3:
                raise ValueError(f"Not an RGB image: {image_array.shape}")

            output_path = os.path.join(output_dir, fname)
            save_uint16_rgb_as_png(image_array, output_path)

        except Exception as e:
            print(f"❌ Failed to process {fname}: {e}")


def save_uint16_rgb_as_png(image_array, path):
    """
    Save 16-bit RGB image by converting to 8-bit before saving as PNG.
    """
    if image_array.dtype == np.uint16:
        max_val = image_array.max()
        if max_val > 255:
            image_array = (image_array / 256).astype(np.uint8)
        else:
            image_array = image_array.astype(np.uint8)

    img = Image.fromarray(image_array, mode='RGB')
    img.save(path)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    run_c5_inference(args)
