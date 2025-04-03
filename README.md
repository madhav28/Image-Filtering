# Image Processing Filters

This project implements various image processing filters including Gaussian Filter, Sobel Operator, and Laplacian of Gaussian (LoG) Filter. The implementation is done in Python using `filters.py`.

## Tasks Completed

### 1. Gaussian Filter
- **Theory**: Proved that a 2D Gaussian convolution can be decomposed into sequential 1D convolutions (horizontal and vertical). The relationship between variances is: the 2D variance is the same as the 1D variance since the Gaussian is separable.
- **Implementation**: 
  - Implemented `convolve()` in `filters.py` with a 3×3 Gaussian kernel (σ² ≈ 1/(2 ln 2)).
  - **Output**: The Gaussian-filtered image (`grace_hopper.png`) shows blurring, which reduces noise and smooths the image.
- **Edge Detection**:
  - Derived kernels for derivatives `kx = [0.5, 0, -0.5]` (1×3) and `ky = [0.5; 0; -0.5]` (3×1).
  - Implemented `edge_detection()` to compute gradient magnitude.
  - **Comparison**: 
    - Original image edge detection shows sharper but noisier edges.
    - Gaussian-filtered edge detection has smoother edges with reduced noise.

### 2. Sobel Operator
- **Theory**: 
  - Proved that derivatives of a Gaussian-filtered image can be approximated using Sobel operators `Gx` and `Gy`.
  - Derived steerable filter kernel `K(α)` for `S(I, α) = Gx cos α + Gy sin α`.
- **Implementation**:
  - Implemented `sobel_operator()` to compute `Gx`, `Gy`, and gradient magnitude.
  - **Output**: `Gx` detects vertical edges, `Gy` detects horizontal edges, and gradient magnitude combines both.
- **Steerable Filter**:
  - Implemented `steerable_filter()` for angles `α = 0, π/6, π/3, π/2, 2π/3, 5π/6`.
  - **Observations**: 
    - The filter detects edges oriented at angle `α`.
    - Outputs rotate with `α`, highlighting edges perpendicular to the direction of `α`.

### 3. LoG Filter
- **Implementation**:
  - Applied two given LoG filters to `grace_hopper.png`.
  - **Output**: 
    - LoG outputs show zero-crossings at edges and blob-like structures.
    - The two filters differ in scale (σ), leading to different blob detection sensitivities.
- **DoG Approximation**:
  - Explained that DoG approximates LoG because the Laplacian of Gaussian is similar to the difference of two Gaussians with close variances (visualizing their shapes shows this resemblance).

## How to Run
1. Place `grace_hopper.png` in the same directory as `filters.py`.
2. Run `python filters.py` to generate outputs for all filters.
3. Output images are saved as:
   - `gaussian_filtered.png`
   - `edge_detection_original.png`
   - `edge_detection_gaussian.png`
   - `sobel_gx.png`, `sobel_gy.png`, `sobel_magnitude.png`
   - `steerable_alpha_{angle}.png` (for each α)
   - `log_filter1.png`, `log_filter2.png`

## Dependencies
- Python 3.x
- NumPy
- OpenCV (`cv2`) or PIL/Pillow
- Matplotlib (for plotting)