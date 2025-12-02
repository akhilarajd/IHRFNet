import os
import scipy.io
import numpy as np
import cv2
from tqdm import tqdm
from natsort import natsorted
from joblib import Parallel, delayed
import argparse

parser = argparse.ArgumentParser(description='Generate patches from Full Resolution images')
parser.add_argument('--src_dir', default='/workspace/SIDD/val', type=str, help='Directory containing .mat files')
parser.add_argument('--tar_dir', default='/workspace/Datasets/val', type=str, help='Directory for image patches')
parser.add_argument('--ps', default=128, type=int, help='Image Patch Size')
parser.add_argument('--num_patches', default=300, type=int, help='Number of patches per image')
parser.add_argument('--num_cores', default=16, type=int, help='Number of CPU Cores')

args = parser.parse_args()

src = args.src_dir
tar = args.tar_dir
PS = args.ps
NUM_PATCHES = args.num_patches
NUM_CORES = args.num_cores

# Paths for patch directories
noisy_patch_dir = os.path.join(tar, 'input')
clean_patch_dir = os.path.join(tar, 'target')

# Create directories if they don't exist
os.makedirs(noisy_patch_dir, exist_ok=True)
os.makedirs(clean_patch_dir, exist_ok=True)

# Automatically find .mat files in the directory
mat_files = [f for f in os.listdir(src) if f.endswith('.mat')]
assert len(mat_files) == 2, f"Expected 2 .mat files, but found {len(mat_files)} files in {src}"

# Load the noisy and ground truth .mat files
noisy_mat_file = [f for f in mat_files if 'noisy' in f.lower()][0]
gt_mat_file = [f for f in mat_files if 'gt' in f.lower()][0]

# Load .mat files
noisy_data = scipy.io.loadmat(os.path.join(src, noisy_mat_file))
gt_data = scipy.io.loadmat(os.path.join(src, gt_mat_file))

# Assuming the data in .mat files are in the shape (40, 32, 256, 256, 3)
noisy_images = noisy_data['ValidationNoisyBlocksSrgb']
gt_images = gt_data['ValidationGtBlocksSrgb']

# Check if the dimensions of both datasets match
assert noisy_images.shape == gt_images.shape, "Shape mismatch between noisy and GT images!"

# Function to save patches
def save_patches(i):
    noisy_img = noisy_images[i]
    clean_img = gt_images[i]
    H, W = noisy_img.shape[0], noisy_img.shape[1]
    
    # Ensure patch size is smaller than image dimensions
    if H < PS or W < PS:
        # If image dimensions are smaller than the patch size, save the entire image directly
        print(f"Image {i+1} has smaller dimensions ({H}x{W}) than the patch size {PS}. Saving the full image directly.")
        
        # Save the full noisy and clean images
        cv2.imwrite(os.path.join(noisy_patch_dir, f'{i+1}.png'), noisy_img)
        cv2.imwrite(os.path.join(clean_patch_dir, f'{i+1}.png'), clean_img)
    else:
        for j in range(NUM_PATCHES):
            # Randomly select the starting coordinates for the patch
            rr = np.random.randint(0, H - PS)
            cc = np.random.randint(0, W - PS)

            # Extract patches
            noisy_patch = noisy_img[rr:rr + PS, cc:cc + PS, :]
            clean_patch = clean_img[rr:rr + PS, cc:cc + PS, :]

            # Save the patches as PNG
            cv2.imwrite(os.path.join(noisy_patch_dir, f'{i+1}_{j+1}.png'), noisy_patch)
            cv2.imwrite(os.path.join(clean_patch_dir, f'{i+1}_{j+1}.png'), clean_patch)


# Use parallel processing to generate patches for all images
Parallel(n_jobs=NUM_CORES)(delayed(save_patches)(i) for i in tqdm(range(noisy_images.shape[0])))

