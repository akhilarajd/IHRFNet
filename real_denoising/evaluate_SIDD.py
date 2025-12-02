import torch
import numpy as np
import scipy.io
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Load data
denoised_data = scipy.io.loadmat('./results/SIDD/mat/Idenoised.mat')['Idenoised']
gt_data = scipy.io.loadmat('./Datasets/SIDD/test/ValidationGtBlocksSrgb.mat')['ValidationGtBlocksSrgb']

# Convert to PyTorch tensors and normalize to range [0, 1]
denoised = torch.tensor(denoised_data, dtype=torch.float32)
gt = torch.tensor(gt_data, dtype=torch.float32)

# Ensure data is in range [0, 1]
denoised = denoised / denoised.max()
gt = gt / gt.max()

# Initialize PSNR and SSIM totals
total_psnr = 0.0
total_ssim = 0.0

# Loop through the patches
for i in range(40):
    for k in range(32):
        denoised_patch = denoised[i, k].numpy()
        gt_patch = gt[i, k].numpy()
        
        # Get patch dimensions
        patch_height, patch_width, _ = gt_patch.shape
        
        # Adjust win_size for SSIM if patch size is small
        win_size = min(7, patch_height, patch_width)
        if win_size % 2 == 0:
            win_size -= 1  # Ensure win_size is odd
        
        # Compute PSNR
        psnr_val = psnr(gt_patch, denoised_patch, data_range=1.0)
        
        # Compute SSIM with adjusted window size
        ssim_val = ssim(
            gt_patch, 
            denoised_patch, 
            data_range=1.0, 
            win_size=win_size, 
            channel_axis=-1
        )
        
        total_psnr += psnr_val
        total_ssim += ssim_val

# Compute average metrics
qm_psnr = total_psnr / (40 * 32)
qm_ssim = total_ssim / (40 * 32)

# Print results
print(f"PSNR: {qm_psnr:.6f} SSIM: {qm_ssim:.6f}")

