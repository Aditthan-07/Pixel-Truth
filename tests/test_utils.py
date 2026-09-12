import unittest
import io
import numpy as np
from PIL import Image
from backend.utils import is_allowed_extension, sanitize_and_prepare_image

class TestBackendUtils(unittest.TestCase):
    def test_allowed_extensions(self):
        self.assertTrue(is_allowed_extension('photo.jpg'))
        self.assertTrue(is_allowed_extension('photo.PNG'))
        self.assertTrue(is_allowed_extension('art.WEBP'))
        self.assertTrue(is_allowed_extension('sample.gif'))
        self.assertFalse(is_allowed_extension('script.py'))
        self.assertFalse(is_allowed_extension('malware.exe'))
        self.assertFalse(is_allowed_extension('no_extension'))
        self.assertFalse(is_allowed_extension(''))

    def test_sanitize_and_prepare_image_valid(self):
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        
        processed = sanitize_and_prepare_image(buf.getvalue())
        self.assertEqual(processed.size, (100, 100))
        self.assertEqual(processed.mode, 'RGB')

    def test_sanitize_and_prepare_image_downscale(self):
        arr = np.zeros((200, 100, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        
        processed = sanitize_and_prepare_image(buf.getvalue(), max_dim=50)
        self.assertLessEqual(max(processed.size), 50)
        self.assertEqual(processed.mode, 'RGB')

if __name__ == '__main__':
    unittest.main()
