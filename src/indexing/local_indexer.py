import sqlite3
import json
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from src.config import settings
from src.video.processor import SceneShot, Keyframe, VideoProcessor
from src.indexing.embeddings import EmbeddingEngine

def parse_query_timestamp_range(query_text: str, video_duration: float = 0.0) -> Optional[Tuple[float, float]]:
    """Parse target timestamp seconds range from natural language queries."""
    # 1. Match HH:MM:SS or MM:SS patterns (e.g. 00:00:16 to 00:00:25)
    hhmmss = re.findall(r'\b(?:\d{1,2}:)?\d{2}:\d{2}\b', query_text)
    if len(hhmmss) >= 1:
        secs = []
        for ts in hhmmss:
            parts = [int(p) for p in ts.split(':')]
            if len(parts) == 3:
                secs.append(parts[0]*3600 + parts[1]*60 + parts[2])
            elif len(parts) == 2:
                secs.append(parts[0]*60 + parts[1])
        if len(secs) == 1:
            return (float(secs[0]), float(secs[0]))
        return (float(min(secs)), float(max(secs)))

    # 2. Match float/int numbers (e.g. "between 16 to 25 seconds" or "0.16 to 0.25")
    nums = [float(n) for n in re.findall(r'\b\d+(?:\.\d+)?\b', query_text)]
    if not nums:
        return None

    # Handle decimal notation like 0.16 to 0.25 when video length > 10s
    if video_duration > 10.0:
        scaled_nums = []
        for n in nums:
            if 0.0 < n < 1.0:
                scaled_nums.append(n * 100.0)
            else:
                scaled_nums.append(n)
        nums = scaled_nums

    if len(nums) == 1:
        return (nums[0], nums[0])
    
    return (min(nums), max(nums))

class LocalIndexer:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or settings.db_path
        self._init_db()
        self.yolo_model = None
        self._load_yolo()

    def _load_yolo(self):
        """Lazy load YOLO model if available."""
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(settings.yolo_model_name)
        except Exception:
            self.yolo_model = None

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize SQLite database tables for video shots, frame vectors, and YOLO tags."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS videos (
                    video_id TEXT PRIMARY KEY,
                    video_path TEXT NOT NULL,
                    duration_sec REAL,
                    shot_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS shots (
                    shot_id TEXT PRIMARY KEY,
                    video_id TEXT NOT NULL,
                    shot_index INTEGER NOT NULL,
                    start_sec REAL NOT NULL,
                    end_sec REAL NOT NULL,
                    start_ts TEXT NOT NULL,
                    end_ts TEXT NOT NULL,
                    storyboard_b64 TEXT,
                    tags_json TEXT,
                    FOREIGN KEY(video_id) REFERENCES videos(video_id)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS keyframes (
                    kf_id TEXT PRIMARY KEY,
                    shot_id TEXT NOT NULL,
                    frame_idx INTEGER NOT NULL,
                    pts_sec REAL NOT NULL,
                    timestamp_str TEXT NOT NULL,
                    clip_vector BLOB,
                    FOREIGN KEY(shot_id) REFERENCES shots(shot_id)
                )
            """)
            conn.commit()

    def index_video(self, video_path: str, shots: List[SceneShot]) -> str:
        """Extract frame vectors and YOLO tags for all shots and persist into SQLite."""
        path = Path(video_path)
        video_id = path.stem
        embed_engine = EmbeddingEngine.get_instance()

        total_duration = shots[-1].end_seconds if shots else 0.0

        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO videos (video_id, video_path, duration_sec, shot_count) VALUES (?, ?, ?, ?)",
                (video_id, str(path.absolute()), total_duration, len(shots))
            )

            for shot in shots:
                shot_db_id = f"{video_id}_shot_{shot.shot_id}"
                
                # Detect YOLO tags from keyframes
                tags = set()
                if self.yolo_model and shot.keyframes:
                    for kf in shot.keyframes:
                        try:
                            results = self.yolo_model(kf.image, verbose=False)
                            for r in results:
                                for c in r.boxes.cls:
                                    tags.add(self.yolo_model.names[int(c)])
                        except Exception:
                            pass

                # Create timestamped storyboard contact sheet base64 payload
                if shot.keyframes:
                    storyboard_img = VideoProcessor.create_storyboard(shot.keyframes)
                    b64_payload = VideoProcessor.storyboard_to_base64(storyboard_img)
                else:
                    b64_payload = ""

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO shots 
                    (shot_id, video_id, shot_index, start_sec, end_sec, start_ts, end_ts, storyboard_b64, tags_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        shot_db_id,
                        video_id,
                        shot.shot_id,
                        shot.start_seconds,
                        shot.end_seconds,
                        shot.start_timestamp,
                        shot.end_timestamp,
                        b64_payload,
                        json.dumps(list(tags))
                    )
                )

                # Compute CLIP visual vector embeddings per keyframe
                for kf in shot.keyframes:
                    kf_db_id = f"{shot_db_id}_kf_{kf.frame_index}"
                    vector = embed_engine.embed_image(kf.image)
                    vector_blob = vector.tobytes()

                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO keyframes
                        (kf_id, shot_id, frame_idx, pts_sec, timestamp_str, clip_vector)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (kf_db_id, shot_db_id, kf.frame_index, kf.pts_seconds, kf.timestamp_str, vector_blob)
                    )

            conn.commit()

        return video_id

    def search_shots(self, video_id: str, query_text: str, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """Perform vector cosine similarity search and timestamp range matching of query_text against video shots."""
        embed_engine = EmbeddingEngine.get_instance()
        query_vector = embed_engine.embed_clip_text(query_text)

        duration = 0.0
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT duration_sec FROM videos WHERE video_id = ?", (video_id,))
            v_row = cursor.fetchone()
            if v_row and v_row["duration_sec"]:
                duration = v_row["duration_sec"]

            cursor.execute(
                """
                SELECT k.shot_id, k.clip_vector, k.timestamp_str, s.*
                FROM keyframes k
                JOIN shots s ON k.shot_id = s.shot_id
                WHERE s.video_id = ?
                """,
                (video_id,)
            )
            rows = cursor.fetchall()

        if not rows:
            return []

        ts_range = parse_query_timestamp_range(query_text, video_duration=duration)

        shot_scores: Dict[str, float] = {}
        shot_data: Dict[str, Dict[str, Any]] = {}

        for row in rows:
            shot_id = row["shot_id"]
            vector_bytes = row["clip_vector"]
            if not vector_bytes:
                continue
            kf_vector = np.frombuffer(vector_bytes, dtype=np.float32)
            sim = embed_engine.cosine_similarity(query_vector, kf_vector)

            start_sec = row["start_sec"]
            end_sec = row["end_sec"]

            # If user specified a timestamp range, boost shots overlapping that range
            if ts_range:
                t_start, t_end = ts_range
                overlap = max(0.0, min(end_sec, t_end) - max(start_sec, t_start))
                if overlap > 0.0 or (t_start == t_end and start_sec <= t_start <= end_sec):
                    sim += 5.0  # Priority boost for matching time interval

            # Check if query matches local YOLO object tags
            tags = json.loads(row["tags_json"] or "[]")
            if any(q_word.lower() in tag.lower() for q_word in query_text.split() for tag in tags):
                sim += 0.15

            if shot_id not in shot_scores or sim > shot_scores[shot_id]:
                shot_scores[shot_id] = sim
                if shot_id not in shot_data:
                    shot_data[shot_id] = {
                        "shot_id": shot_id,
                        "video_id": row["video_id"],
                        "shot_index": row["shot_index"],
                        "start_sec": row["start_sec"],
                        "end_sec": row["end_sec"],
                        "start_ts": row["start_ts"],
                        "end_ts": row["end_ts"],
                        "storyboard_b64": row["storyboard_b64"],
                        "tags": tags,
                        "kf_vectors": []
                    }
                shot_data[shot_id]["kf_vectors"].append(kf_vector)

        sorted_shots = sorted(shot_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(shot_data[s_id], score) for s_id, score in sorted_shots]
