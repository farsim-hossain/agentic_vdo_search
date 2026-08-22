import unittest
from llama_index.core.schema import ImageNode
from src.llamaindex.embeddings import ClipTextEmbedding, SentenceTransformerTextEmbedding
from src.llamaindex.index_builder import LlamaVideoIndexBuilder

class TestLlamaIndexAdapter(unittest.TestCase):
    def test_clip_text_embedding(self):
        embed_model = ClipTextEmbedding()
        vec = embed_model.get_text_embedding("person walking near car")
        self.assertIsInstance(vec, list)
        self.assertEqual(len(vec), 512)

    def test_sentence_transformer_embedding(self):
        embed_model = SentenceTransformerTextEmbedding()
        vec = embed_model.get_text_embedding("video summary")
        self.assertIsInstance(vec, list)
        self.assertEqual(len(vec), 384)

    def test_node_builder_creation(self):
        builder = LlamaVideoIndexBuilder()
        self.assertIsNotNone(builder)

if __name__ == "__main__":
    unittest.main()
