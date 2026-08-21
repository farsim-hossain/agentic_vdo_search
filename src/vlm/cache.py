import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from src.config import settings
from src.indexing.embeddings import EmbeddingEngine

class VLMCache:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or settings.db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vlm_observations (
                    shot_id TEXT PRIMARY KEY,
                    video_id TEXT NOT NULL,
                    raw_json TEXT NOT NULL,
                    summary_text TEXT NOT NULL,
                    text_vector BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def get_observation(self, shot_id: str) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM vlm_observations WHERE shot_id = ?", (shot_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "shot_id": row["shot_id"],
                "video_id": row["video_id"],
                "raw_json": json.loads(row["raw_json"]),
                "summary_text": row["summary_text"],
                "created_at": row["created_at"]
            }

    def save_observation(self, shot_id: str, video_id: str, raw_json_data: Dict[str, Any], summary_text: str):
        embed_engine = EmbeddingEngine.get_instance()
        text_vec = embed_engine.embed_text(summary_text)
        vec_bytes = text_vec.tobytes()

        raw_json_str = json.dumps(raw_json_data)

        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO vlm_observations
                (shot_id, video_id, raw_json, summary_text, text_vector)
                VALUES (?, ?, ?, ?, ?)
                """,
                (shot_id, video_id, raw_json_str, summary_text, vec_bytes)
            )
            conn.commit()

    def search_observations(self, video_id: str, query_text: str, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        embed_engine = EmbeddingEngine.get_instance()
        query_vec = embed_engine.embed_text(query_text)

        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM vlm_observations WHERE video_id = ?", (video_id,))
            rows = cursor.fetchall()

        if not rows:
            return []

        results = []
        for row in rows:
            vec_bytes = row["text_vector"]
            if not vec_bytes:
                continue
            doc_vec = np.frombuffer(vec_bytes, dtype=np.float32)
            sim = embed_engine.cosine_similarity(query_vec, doc_vec)

            obs_data = {
                "shot_id": row["shot_id"],
                "video_id": row["video_id"],
                "raw_json": json.loads(row["raw_json"]),
                "summary_text": row["summary_text"],
            }
            results.append((obs_data, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
