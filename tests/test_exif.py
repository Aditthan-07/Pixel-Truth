import unittest
import io
from PIL import Image, PngImagePlugin
from backend.exif_inspector import inspect_image_metadata

class TestExifInspector(unittest.TestCase):
    def test_plain_image_no_exif(self):
        img = Image.new('RGB', (50, 50), color='blue')
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        
        info = inspect_image_metadata(buf.getvalue())
        self.assertFalse(info['has_camera_data'])
        self.assertFalse(info['ai_metadata_detected'])

    def test_png_ai_prompt_metadata(self):
        img = Image.new('RGB', (50, 50), color='green')
        meta = PngImagePlugin.PngInfo()
        meta.add_text('parameters', 'A beautiful photo, Stable Diffusion XL, steps: 30')
        buf = io.BytesIO()
        img.save(buf, format='PNG', pnginfo=meta)

        info = inspect_image_metadata(buf.getvalue())
        self.assertTrue(info['ai_metadata_detected'])
        self.assertEqual(info['ai_software_tag'], 'parameters')
        self.assertIn('parameters', info['raw_text_chunks'])

    def test_corrupted_bytes_handling(self):
        info = inspect_image_metadata(b'corrupted_bytes_123')
        self.assertFalse(info['has_exif'])
        self.assertFalse(info['ai_metadata_detected'])

if __name__ == '__main__':
    unittest.main()
