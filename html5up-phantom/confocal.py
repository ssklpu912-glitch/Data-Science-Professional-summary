"""Confocal / Fluorescence classical image-processing pipeline for LabLogbook.

Covers the pre-segmentation part of a real confocal workflow using only
scikit-image/scipy/numpy — no ML segmentation here by design (Cellpose/StarDist
are a separate, deliberately-deferred step):

  - Background subtraction (rolling-ball)
  - Flat-field / illumination correction (self-estimated, since we don't have a
    separate calibration/blank frame — divides by a heavily-blurred version of
    the image itself)
  - Denoising (Gaussian or median)
  - Deconvolution (Richardson-Lucy against a synthetic Gaussian PSF, since we
    don't have a measured PSF for the instrument that took the image)
  - Channel alignment/registration (phase cross-correlation + subpixel shift)
  - Classical segmentation (Otsu threshold + distance-transform watershed, to
    separate touching objects) with per-object area/intensity stats

Honesty notes (same policy as the rest of the app): the flat-field correction
here is a self-estimated illumination field, not a true calibration against a
blank/reference frame — good for comparative use, not absolute photometry. The
deconvolution PSF is a synthetic Gaussian approximation, not the instrument's
measured PSF — it sharpens but isn't a substitute for a calibrated deconvolution.
"""
import os
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import shift as ndi_shift, distance_transform_edt
from skimage.filters import gaussian, median, threshold_otsu
from skimage.morphology import disk, remove_small_objects, remove_small_holes
from skimage.restoration import rolling_ball, richardson_lucy
from skimage.registration import phase_cross_correlation
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
from skimage.color import label2rgb

PREPROCESS_STEPS = ('background', 'flatfield', 'denoise', 'deconvolve')

DENOISE_METHODS = ('gaussian', 'median')


def _timestamp_filename(prefix, tag=''):
    tag_part = f"{tag}_" if tag else ''
    return f"{prefix}_{tag_part}{int(datetime.now().timestamp() * 1000)}.png"


def subtract_background(gray, radius=25):
    """Rolling-ball background estimate, subtracted and clipped at zero."""
    bg = rolling_ball(gray, radius=radius)
    corrected = gray - bg
    corrected[corrected < 0] = 0
    return corrected, bg


def flatfield_correct(gray, sigma=50):
    """Self-estimated illumination correction: divide by a heavily-blurred
    copy of the image itself (no separate blank/calibration frame available)."""
    illumination = gaussian(gray, sigma=sigma, preserve_range=True)
    illumination = np.where(illumination <= 1e-6, 1e-6, illumination)
    corrected = gray / illumination * np.mean(illumination)
    return corrected


def denoise_image(gray, method='gaussian', amount=1.0):
    if method == 'median':
        radius = max(1, int(round(amount)))
        return median(gray, disk(radius))
    sigma = max(0.1, float(amount))
    return gaussian(gray, sigma=sigma, preserve_range=True)


def deconvolve(gray, psf_sigma=2.0, iterations=15):
    """Richardson-Lucy deconvolution against a synthetic Gaussian PSF — an
    approximation used because the instrument's real measured PSF isn't
    available from a plain image file."""
    size = max(5, int(psf_sigma * 6) | 1)
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx ** 2 + yy ** 2) / (2 * psf_sigma ** 2))
    psf /= psf.sum()

    peak = float(gray.max()) or 1.0
    normalized = np.clip(gray, 0, None) / peak
    deconvolved = richardson_lucy(normalized, psf, num_iter=int(iterations), clip=False)
    return np.clip(deconvolved, 0, None) * peak


def run_preprocess(gray, steps, params):
    """Applies the requested steps in a fixed, sensible order (background ->
    flat-field -> denoise -> deconvolve) and returns (result_array, log_lines)."""
    result = gray.astype(float).copy()
    log = []

    if 'background' in steps:
        radius = params.get('bg_radius', 25)
        result, _ = subtract_background(result, radius=radius)
        log.append(f"Background subtracted (rolling-ball, radius={radius}px)")

    if 'flatfield' in steps:
        sigma = params.get('flatfield_sigma', 50)
        result = flatfield_correct(result, sigma=sigma)
        log.append(f"Flat-field corrected (self-estimated illumination, sigma={sigma}px)")

    if 'denoise' in steps:
        method = params.get('denoise_method', 'gaussian')
        amount = params.get('denoise_amount', 1.0)
        result = denoise_image(result, method=method, amount=amount)
        log.append(f"Denoised ({method}, amount={amount})")

    if 'deconvolve' in steps:
        psf_sigma = params.get('psf_sigma', 2.0)
        iterations = params.get('deconv_iterations', 15)
        result = deconvolve(result, psf_sigma=psf_sigma, iterations=iterations)
        log.append(f"Deconvolved (Richardson-Lucy, synthetic Gaussian PSF sigma={psf_sigma}px, {iterations} iterations)")

    return result, log


def align_channels(reference_gray, moving_gray):
    """Subpixel registration via cross-correlation; returns the aligned moving
    image, the (dy, dx) shift applied, and the registration error.

    Uses unnormalized cross-correlation (normalization=None) rather than the
    phase-only default: phase normalization is more sensitive to noise and,
    in this skimage version, its reported error saturates at ~1.0 regardless
    of match quality — not useful for judging alignment quality on noisy
    fluorescence images. Unnormalized correlation gives a genuinely
    informative near-zero error for a good match."""
    shift_yx, error, _phase = phase_cross_correlation(
        reference_gray, moving_gray, upsample_factor=10, normalization=None)
    aligned = ndi_shift(moving_gray, shift=shift_yx, mode='constant', cval=0)
    return aligned, shift_yx, float(error)


def segment_watershed(gray, min_distance=10, threshold_offset=0.0, min_object_size=9):
    """Otsu threshold + distance-transform watershed, splitting touching
    objects into separate labeled regions. Returns (labels_image, mask,
    per_object_stats)."""
    smoothed = gaussian(gray, sigma=1.0, preserve_range=True)
    thresh = threshold_otsu(smoothed) + threshold_offset

    mask = smoothed > thresh
    mask = remove_small_objects(mask, max_size=min_object_size)
    mask = remove_small_holes(mask, max_size=min_object_size)

    distance = distance_transform_edt(mask)
    coords = peak_local_max(distance, min_distance=max(1, int(min_distance)), labels=mask)
    seed_mask = np.zeros(distance.shape, dtype=bool)
    if len(coords):
        seed_mask[tuple(coords.T)] = True
    markers = label(seed_mask)

    labels_image = watershed(-distance, markers, mask=mask)

    objects = []
    for r in regionprops(labels_image, intensity_image=gray):
        objects.append({
            'label': int(r.label),
            'area_px': int(r.area),
            'equiv_diameter_px': float(r.equivalent_diameter_area),
            'mean_intensity': float(r.intensity_mean),
            'centroid': (float(r.centroid[0]), float(r.centroid[1])),
        })

    return labels_image, mask, objects


# ---------------------------------------------------------------------------
# Plotting helpers — each saves a PNG to out_dir and returns its filename,
# mirroring how the rest of the app (generate_porosity_overlay etc.) works.
# ---------------------------------------------------------------------------

def render_preprocess_comparison(original, processed, out_dir, tag=''):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    axes[1].imshow(processed, cmap='gray')
    axes[1].set_title('Processed')
    axes[1].axis('off')
    fig.tight_layout()

    filename = _timestamp_filename('confocal_preprocess', tag)
    fig.savefig(os.path.join(out_dir, filename), dpi=130)
    plt.close(fig)
    return filename


def render_alignment_comparison(reference, moving, aligned, shift_yx, out_dir, tag=''):
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, arr, title in zip(
        axes, [reference, moving, aligned],
        ['Reference channel', 'Moving channel (before)', f'Moving channel (aligned, shift={shift_yx[0]:.2f},{shift_yx[1]:.2f}px)'],
    ):
        ax.imshow(arr, cmap='gray')
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    fig.tight_layout()

    filename = _timestamp_filename('confocal_align', tag)
    fig.savefig(os.path.join(out_dir, filename), dpi=130)
    plt.close(fig)
    return filename


def render_segmentation_overlay(gray, labels_image, out_dir, tag=''):
    gray_norm = gray - gray.min()
    peak = gray_norm.max() or 1.0
    gray_norm = gray_norm / peak
    overlay = label2rgb(labels_image, image=gray_norm, bg_label=0, alpha=0.4, image_alpha=1.0)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(overlay)
    ax.axis('off')
    n_objects = int(labels_image.max())
    ax.set_title(f"{n_objects} object{'s' if n_objects != 1 else ''} detected")
    fig.tight_layout()

    filename = _timestamp_filename('confocal_segment', tag)
    fig.savefig(os.path.join(out_dir, filename), dpi=130)
    plt.close(fig)
    return filename
