import unittest
import numpy as np
from PIL import Image

def normalize_label(label: str) -> str:
    l = label.lower()
    if any(k in l for k in ('artificial', 'ai', 'fake')):
        return 'AI-Generated'
    if any(k in l for k in ('human', 'real', 'photo')):
        return 'Real-Image'
    return label

def vote_and_average_fusion(model_outputs):
    S_AI, S_real = 0.0, 0.0
    V_AI, V_real = 0, 0
    n = len(model_outputs)

    for results in model_outputs:
        for r in results:
            bucket = normalize_label(r['label'])
            score = r['score']
            if bucket == 'AI-Generated':
                S_AI += score
                if score > 0.5:
                    V_AI += 1
            elif bucket == 'Real-Image':
                S_real += score
                if score > 0.5:
                    V_real += 1

    A_AI = S_AI / n if n > 0 else 0.0
    A_real = S_real / n if n > 0 else 0.0

    if V_AI > V_real:
        return 'AI-Generated', round(A_AI * 100, 2)
    elif V_real > V_AI:
        return 'Real-Image', round(A_real * 100, 2)
    elif A_AI >= A_real:
        return 'AI-Generated', round(A_AI * 100, 2)
    else:
        return 'Real-Image', round(A_real * 100, 2)

class TestPixelTruthFusion(unittest.TestCase):
    def test_label_normalization(self):
        self.assertEqual(normalize_label("artificial_art"), "AI-Generated")
        self.assertEqual(normalize_label("ai_photo"), "AI-Generated")
        self.assertEqual(normalize_label("fake"), "AI-Generated")
        self.assertEqual(normalize_label("real_photograph"), "Real-Image")
        self.assertEqual(normalize_label("human_portrait"), "Real-Image")

    def test_both_models_agree_ai(self):
        outputs = [
            [{'label': 'ai', 'score': 0.91}],
            [{'label': 'artificial', 'score': 0.88}]
        ]
        label, conf = vote_and_average_fusion(outputs)
        self.assertEqual(label, "AI-Generated")
        self.assertAlmostEqual(conf, 89.5, places=1)

    def test_both_models_agree_real(self):
        outputs = [
            [{'label': 'real', 'score': 0.95}],
            [{'label': 'photo', 'score': 0.90}]
        ]
        label, conf = vote_and_average_fusion(outputs)
        self.assertEqual(label, "Real-Image")
        self.assertAlmostEqual(conf, 92.5, places=1)

    def test_tie_break_resolution(self):
        outputs = [
            [{'label': 'ai', 'score': 0.83}],
            [{'label': 'real', 'score': 0.31}]
        ]
        label, conf = vote_and_average_fusion(outputs)
        self.assertEqual(label, "AI-Generated")

    def test_synthetic_image_tensor_creation(self):
        img_arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_arr)
        self.assertEqual(img.size, (224, 224))
        self.assertEqual(img.mode, 'RGB')

if __name__ == '__main__':
    unittest.main()
