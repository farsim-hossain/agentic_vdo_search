import numpy as np
from typing import List, Union, Optional
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from src.config import settings

class EmbeddingEngine:
    _instance: Optional["EmbeddingEngine"] = None

    def __init__(self):
        self.device = "cpu"
        
        # Load CLIP for image & visual text search
        self.clip_processor = CLIPProcessor.from_pretrained(settings.clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(settings.clip_model_name).to(self.device)
        self.clip_model.eval()

        # Load SentenceTransformer for text-to-text semantic VLM observation search
        self.text_model = SentenceTransformer(settings.text_embed_model_name, device=self.device)

    @classmethod
    def get_instance(cls) -> "EmbeddingEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Embed PIL Image into a 512-dim normalized float32 CLIP vector."""
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            # Normalize vector
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            return image_features.cpu().numpy().flatten().astype(np.float32)

    def embed_clip_text(self, text: str) -> np.ndarray:
        """Embed text query into 512-dim normalized float32 CLIP vector space for visual matching."""
        inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            return text_features.cpu().numpy().flatten().astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        """Embed text query into 384-dim normalized float32 SentenceTransformer vector for VLM text search."""
        vector = self.text_model.encode(text, normalize_embeddings=True)
        return np.array(vector, dtype=np.float32)

    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine similarity between two normalized float vectors."""
        if v1.ndim == 1:
            v1 = v1.reshape(1, -1)
        if v2.ndim == 1:
            v2 = v2.reshape(1, -1)
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
        if denom == 0:
            return 0.0
        return float(np.dot(v1, v2.T)[0, 0] / denom)
