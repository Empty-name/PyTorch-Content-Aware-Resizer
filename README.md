# PyTorch Content-Aware Resizer (Seam Carving)

A high-performance PyTorch implementation of the Seam Carving algorithm for content-aware image resizing. Unlike naive scaling or cropping, this engine dynamically identifies and removes (or duplicates) the optimal paths of least visual importance ("seams") using energy maps and dynamic programming.

## Visual Results
![Original Image](assets/couple.jpg) 
![Seam Inserted](assets/couple_inserted.jpg)

## Technical Highlights
* **Tensor Vectorization:** Replaced slow iterative loops with PyTorch boolean masking (`img[mask]`) and dynamic reshaping for rapid pixel removal.
* **Dynamic Programming:** Efficiently computes optimal vertical seams based on Sobel filter energy maps.
* **Coordinate Mapping Engine:** Implemented complex reverse index tracking to allow parallel seam insertion without distorting image geometry, avoiding common "thick line" artifacts.
* **GPU Acceleration:** Fully compatible with CUDA for parallel computation of convolution filters on large images.

## Quick Start
```bash
git clone https://github.com/Empty-name/PyTorch-Content-Aware-Resizer.git
cd PyTorch-Content-Aware-Resizer
pip install -r requirements.txt
