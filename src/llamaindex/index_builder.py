import json
import numpy as np
from typing import List, Dict, Any, Optional
from llama_index.core.schema import ImageNode, TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from src.indexing.local_indexer import LocalIndexer
from src.llamaindex.embeddings import ClipTextEmbedding
from src.llamaindex.multimodal_llm import GroqLlamaMultiModalLLM

class LlamaVideoIndexBuilder:
    """Constructs LlamaIndex ImageNode and TextNode objects for video shots and keyframes."""

    def __init__(self, local_indexer: Optional[LocalIndexer] = None):
        self.indexer = local_indexer or LocalIndexer()
        self.embed_model = ClipTextEmbedding()
        self.multi_modal_llm = GroqLlamaMultiModalLLM()

    def build_nodes_for_video(self, video_id: str) -> List[ImageNode]:
        """Convert video shots stored in SQLite into LlamaIndex ImageNode objects."""
        nodes: List[ImageNode] = []
        with self.indexer._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT s.*, k.clip_vector
                FROM shots s
                LEFT JOIN keyframes k ON s.shot_id = k.shot_id
                WHERE s.video_id = ?
                """,
                (video_id,)
            )
            rows = cursor.fetchall()

        seen_shots = set()
        for row in rows:
            shot_id = row["shot_id"]
            if shot_id in seen_shots:
                continue
            seen_shots.add(shot_id)

            tags = json.loads(row["tags_json"] or "[]")
            b64_img = row["storyboard_b64"] or ""

            extra_info = {
                "shot_id": shot_id,
                "video_id": row["video_id"],
                "shot_index": row["shot_index"],
                "start_sec": row["start_sec"],
                "end_sec": row["end_sec"],
                "start_ts": row["start_ts"],
                "end_ts": row["end_ts"],
                "tags": tags,
            }

            node = ImageNode(
                id_=shot_id,
                image_url=b64_img,
                text=f"Video shot {shot_id} [{row['start_ts']}-{row['end_ts']}]. Objects: {', '.join(tags)}",
                extra_info=extra_info
            )

            # Attach pre-computed CLIP visual vector if present
            vec_bytes = row["clip_vector"]
            if vec_bytes:
                kf_vec = np.frombuffer(vec_bytes, dtype=np.float32)
                node.embedding = kf_vec.tolist()

            nodes.append(node)

        return nodes

    def create_vector_index(self, video_id: str) -> VectorStoreIndex:
        """Create a LlamaIndex VectorStoreIndex for a specific video."""
        nodes = self.build_nodes_for_video(video_id)
        index = VectorStoreIndex(
            nodes=nodes,
            embed_model=self.embed_model,
        )
        return index
