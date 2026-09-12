import unittest
from pixeltruth import PredictionResult, PixelTruthDetector

class TestPixelTruthSDK(unittest.TestCase):
    def test_prediction_result_ai(self):
        res = PredictionResult(label='AI-Generated', confidence=88.5)
        self.assertTrue(res.is_ai)
        self.assertFalse(res.is_real)
        data = res.to_dict()
        self.assertEqual(data['label'], 'AI-Generated')
        self.assertEqual(data['confidence'], 88.5)

    def test_prediction_result_real(self):
        res = PredictionResult(label='Real-Image', confidence=95.2)
        self.assertFalse(res.is_ai)
        self.assertTrue(res.is_real)

    def test_detector_missing_file_raises(self):
        detector = PixelTruthDetector()
        with self.assertRaises(FileNotFoundError):
            detector.predict('nonexistent_file_xyz.jpg')

    def test_detector_invalid_type_raises(self):
        detector = PixelTruthDetector()
        with self.assertRaises(TypeError):
            detector.predict(12345)

if __name__ == '__main__':
    unittest.main()
