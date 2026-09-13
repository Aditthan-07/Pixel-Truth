import unittest
import numpy as np
from backend.gradcam import apply_colormap

class TestGradcamColormap(unittest.TestCase):
    def test_thermal_colormap_shape(self):
        heatmap = np.zeros((32, 32), dtype=np.uint8)
        colored = apply_colormap(heatmap, style='thermal')
        self.assertEqual(colored.shape, (32, 32, 3))
        self.assertEqual(colored.dtype, np.uint8)

    def test_coolwarm_colormap_shape(self):
        heatmap = np.full((32, 32), 128, dtype=np.uint8)
        colored = apply_colormap(heatmap, style='coolwarm')
        self.assertEqual(colored.shape, (32, 32, 3))

    def test_fire_colormap_shape(self):
        heatmap = np.full((32, 32), 255, dtype=np.uint8)
        colored = apply_colormap(heatmap, style='fire')
        self.assertEqual(colored.shape, (32, 32, 3))

if __name__ == '__main__':
    unittest.main()
