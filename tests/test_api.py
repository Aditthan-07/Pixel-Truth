import unittest
import io
from backend.app import app

class TestPixelTruthAPI(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get('status'), 'healthy')
        self.assertIn('PixelTruth', data.get('service', ''))
        self.assertIn('models_loaded', data)

    def test_security_headers(self):
        response = self.client.get('/health')
        self.assertEqual(response.headers.get('X-Content-Type-Options'), 'nosniff')
        self.assertEqual(response.headers.get('X-Frame-Options'), 'DENY')
        self.assertEqual(response.headers.get('Referrer-Policy'), 'no-referrer')

    def test_predict_missing_file(self):
        response = self.client.post('/predict', data={})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn('error', data)
        self.assertIn('No image file provided', data['error'])

    def test_predict_empty_filename(self):
        data = {'image': (io.BytesIO(b'dummy content'), '')}
        response = self.client.post('/predict', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        res_json = response.get_json()
        self.assertIn('error', res_json)
        self.assertIn('No file selected', res_json['error'])

    def test_predict_unsupported_extension(self):
        data = {'image': (io.BytesIO(b'dummy content'), 'test.pdf')}
        response = self.client.post('/predict', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        res_json = response.get_json()
        self.assertIn('error', res_json)
        self.assertIn('Unsupported file type', res_json['error'])

if __name__ == '__main__':
    unittest.main()
