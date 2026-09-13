import numpy as np
from PIL import Image
from typing import Dict, Any

def analyze_frequency_domain(image: Image.Image) -> Dict[str, Any]:
    """
    Analyzes 2D Fourier transform power spectrum to detect high-frequency grid
    anomalies commonly left by diffusion upsamplers and convolutional generators.
    """
    # 1. Convert to grayscale luminance
    gray = image.convert('L')
    arr = np.array(gray, dtype=np.float32)

    # 2. 2D Fast Fourier Transform and zero-frequency centering
    f_transform = np.fft.fft2(arr)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)
    power_spectrum = magnitude ** 2

    # 3. Create radial frequency masks (Low vs High frequency)
    rows, cols = arr.shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    distances = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
    max_radius = np.sqrt(crow ** 2 + ccol ** 2)

    cutoff = 0.5 * max_radius
    high_mask = distances >= cutoff
    low_mask = distances < cutoff

    low_freq_power = float(np.sum(power_spectrum[low_mask]))
    high_freq_power = float(np.sum(power_spectrum[high_mask]))
    total_power = low_freq_power + high_freq_power

    high_freq_ratio = (high_freq_power / total_power) if total_power > 0 else 0.0

    # 4. Spectral entropy estimation (spread of spectral energy)
    norm_power = power_spectrum / (np.sum(power_spectrum) + 1e-12)
    norm_power = norm_power[norm_power > 0]
    spectral_entropy = float(-np.sum(norm_power * np.log2(norm_power + 1e-12)))

    # 5. Radial power variance in high-frequencies
    high_freq_vals = power_spectrum[high_mask]
    high_freq_variance = float(np.var(high_freq_vals)) if len(high_freq_vals) > 0 else 0.0

    return {
        'high_freq_energy_ratio': round(high_freq_ratio, 4),
        'high_freq_variance': round(high_freq_variance, 4),
        'spectral_entropy': round(spectral_entropy, 2),
        'spectrum_dimensions': [rows, cols]
    }
