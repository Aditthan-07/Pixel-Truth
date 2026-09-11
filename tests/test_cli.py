import unittest
import subprocess
import sys
import json
import os

class TestPixelTruthCLI(unittest.TestCase):
    def test_cli_help(self):
        result = subprocess.run(
            [sys.executable, "cli.py", "--help"],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("PixelTruth CLI", result.stdout)
        self.assertIn("--json", result.stdout)
        self.assertIn("--output", result.stdout)

    def test_cli_missing_file_text_mode(self):
        result = subprocess.run(
            [sys.executable, "cli.py", "nonexistent_image_12345.jpg"],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("[ERROR] File not found", result.stderr)

    def test_cli_missing_file_json_mode(self):
        result = subprocess.run(
            [sys.executable, "cli.py", "nonexistent_image_12345.jpg", "--json"],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 1)
        data = json.loads(result.stderr.strip())
        self.assertIn("error", data)
        self.assertIn("File not found", data["error"])

if __name__ == '__main__':
    unittest.main()
