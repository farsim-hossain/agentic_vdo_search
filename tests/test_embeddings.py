import unittest
import numpy as np
from src.indexing.embeddings import EmbeddingEngine

class TestEmbeddings(unittest.TestCase):
    def test_cosine_similarity(self):
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        self.assertAlmostEqual(EmbeddingEngine.cosine_similarity(v1, v2), 1.0, places=5)
        self.assertAlmostEqual(EmbeddingEngine.cosine_similarity(v1, v3), 0.0, places=5)

if __name__ == "__main__":
    unittest.main()
