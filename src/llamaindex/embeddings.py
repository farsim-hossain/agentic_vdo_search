from typing import Any, List
from llama_index.core.embeddings import BaseEmbedding
from src.indexing.embeddings import EmbeddingEngine

class ClipTextEmbedding(BaseEmbedding):
    """LlamaIndex BaseEmbedding adapter for CLIP (openai/clip-vit-base-patch32) text embeddings."""
    
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def _get_text_embedding(self, text: str) -> List[float]:
        embed_engine = EmbeddingEngine.get_instance()
        vec = embed_engine.embed_clip_text(text)
        return vec.tolist()

    def _get_query_embedding(self, query: str) -> List[float]:
        embed_engine = EmbeddingEngine.get_instance()
        vec = embed_engine.embed_clip_text(query)
        return vec.tolist()

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)


class SentenceTransformerTextEmbedding(BaseEmbedding):
    """LlamaIndex BaseEmbedding adapter for SentenceTransformers (all-MiniLM-L6-v2) text embeddings."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def _get_text_embedding(self, text: str) -> List[float]:
        embed_engine = EmbeddingEngine.get_instance()
        vec = embed_engine.embed_text(text)
        return vec.tolist()

    def _get_query_embedding(self, query: str) -> List[float]:
        embed_engine = EmbeddingEngine.get_instance()
        vec = embed_engine.embed_text(query)
        return vec.tolist()

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)
