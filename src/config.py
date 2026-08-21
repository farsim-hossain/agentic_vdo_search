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

    # Groq API Configuration
    groq_api_key: str = Field(default="", env="GROQ_API_KEY")
    groq_vlm_model: str = Field(default="qwen/qwen3.6-27b", env="GROQ_VLM_MODEL")
    groq_text_model: str = Field(default="llama-3.3-70b-versatile", env="GROQ_TEXT_MODEL")
    
    # Rate Limiting
    vlm_min_request_interval: float = Field(default=60.0, description="Minimum seconds between Groq VLM API requests")
    
    # Machine Learning Models (CPU)
    clip_model_name: str = Field(default="openai/clip-vit-base-patch32", description="HuggingFace CLIP model for frame embeddings")
    text_embed_model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="SentenceTransformers model for text embeddings")
    yolo_model_name: str = Field(default="yolov8n.pt", description="Ultralytics YOLO nano model for local object tagging")
    
    # Storage Paths
    db_path: Path = Field(default=Path("video_index.db"), description="SQLite database path for index and VLM cache")
    cache_dir: Path = Field(default=Path(".cache_storyboards"), description="Directory for storyboard contact sheet thumbnails")

settings = Settings()

# Ensure cache directory exists
settings.cache_dir.mkdir(parents=True, exist_ok=True)
