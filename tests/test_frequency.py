import unittest
import numpy as np
from PIL import Image
from backend.frequency import analyze_frequency_domain

class TestFrequencyDomain(unittest.TestCase):
    def test_frequency_metrics_structure(self):
        arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        
        metrics = analyze_frequency_domain(img)
        self.assertIn('high_freq_energy_ratio', metrics)
        self.assertIn('high_freq_variance', metrics)
        self.assertIn('spectral_entropy', metrics)
        self.assertEqual(metrics['spectrum_dimensions'], [64, 64])

    def test_smooth_image_low_frequency(self):
        # A completely flat image has all energy at DC (zero frequency)
        arr = np.full((64, 64), 128, dtype=np.uint8)
        img = Image.fromarray(arr)
        
        metrics = analyze_frequency_domain(img)
        self.assertEqual(metrics['high_freq_energy_ratio'], 0.0)

if __name__ == '__main__':
    unittest.main()
