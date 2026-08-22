import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Groq API Configuration (supports GROQ_VLM_MODEL or GROQ_VLM)
    groq_api_key: str = Field(default="", env="GROQ_API_KEY")
    groq_vlm_model: str = Field(default="qwen/qwen3.6-27b")
    groq_text_model: str = Field(default="openai/gpt-oss-120b")
    
    # Rate Limiting
    vlm_min_request_interval: float = Field(default=60.0, description="Minimum seconds between Groq VLM API requests")
    
    # Machine Learning Models (CPU)
    clip_model_name: str = Field(default="openai/clip-vit-base-patch32", description="HuggingFace CLIP model for frame embeddings")
    text_embed_model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="SentenceTransformers model for text embeddings")
    yolo_model_name: str = Field(default="yolov8n.pt", description="Ultralytics YOLO nano model for local object tagging")
    
    # Storage Paths
    db_path: Path = Field(default=Path("video_index.db"), description="SQLite database path for index and VLM cache")
    cache_dir: Path = Field(default=Path(".cache_storyboards"), description="Directory for storyboard contact sheet thumbnails")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Check env aliases for GROQ_VLM / GROQ_VLM_MODEL
        env_vlm = os.getenv("GROQ_VLM_MODEL") or os.getenv("GROQ_VLM")
        if env_vlm:
            self.groq_vlm_model = env_vlm
        env_text = os.getenv("GROQ_TEXT_MODEL") or os.getenv("GROQ_TEXT") or self.groq_vlm_model
        if env_text:
            self.groq_text_model = env_text

settings = Settings()

# Ensure cache directory exists
settings.cache_dir.mkdir(parents=True, exist_ok=True)
