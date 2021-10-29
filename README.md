# metric_4_talking_heads
## Prerequisites
environment.yaml
## File structure
1. (ground truth images included) folder
2. (reconstructed images included) folder

Images in each 1. and 2. must be sorted w/ same index.
## Usage

python evaluation.py --metric [metric_name]

('akd', 'aed', 'l1', 'psnr', 'ssim', 'ms-ssim' are provided)
