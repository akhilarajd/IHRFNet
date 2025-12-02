import torch
import scipy.io
import piq
import pyiqa  # pip install pyiqa

# --- Load data ---
denoised_data = scipy.io.loadmat('./results/SIDD/mat/Idenoised.mat')['Idenoised']
gt_data = scipy.io.loadmat('./Datasets/SIDD/test/ValidationGtBlocksSrgb.mat')['ValidationGtBlocksSrgb']

# Convert to torch tensors and permute to [N, C, H, W]
denoised = torch.tensor(denoised_data, dtype=torch.float32).permute(0, 1, 4, 2, 3)
gt = torch.tensor(gt_data, dtype=torch.float32).permute(0, 1, 4, 2, 3)

# Normalize to [0, 1]
denoised = denoised / denoised.max()
gt = gt / gt.max()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
denoised = denoised.to(device)
gt = gt.to(device)

# Initialize NIQE and BRISQUE models
niqe_model = pyiqa.create_metric('niqe').to(device)
brisque_model = pyiqa.create_metric('brisque').to(device)

# Initialize totals
total_psnr, total_ssim, total_brisque, total_niqe = 0.0, 0.0, 0.0, 0.0
count = 0

# --- Loop over patches ---
for i in range(denoised.shape[0]):      # 40
    for k in range(denoised.shape[1]):  # 32
        den_patch = denoised[i, k].unsqueeze(0)  # [1, 3, H, W]
        gt_patch = gt[i, k].unsqueeze(0)

        # PSNR & SSIM (reference-based)
        psnr_val = piq.psnr(den_patch, gt_patch, data_range=1.0).item()
        ssim_val = piq.ssim(den_patch, gt_patch, data_range=1.0).item()

        # BRISQUE (no-reference)
        brisque_val = brisque_model(den_patch).item()

        # NIQE (no-reference)
        niqe_val = niqe_model(den_patch).item()

        total_psnr += psnr_val
        total_ssim += ssim_val
        total_brisque += brisque_val
        total_niqe += niqe_val
        count += 1

# --- Compute averages ---
avg_psnr = total_psnr / count
avg_ssim = total_ssim / count
avg_brisque = total_brisque / count
avg_niqe = total_niqe / count

print(f"PSNR: {avg_psnr:.6f}  SSIM: {avg_ssim:.6f}")
print(f"BRISQUE: {avg_brisque:.6f}  NIQE: {avg_niqe:.6f}")
