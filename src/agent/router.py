import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple
from src.video.processor import VideoProcessor
from src.indexing.local_indexer import LocalIndexer
from src.vlm.client import GroqVLMClient
from src.vlm.cache import VLMCache

class AgenticRouter:
    def __init__(self, api_key: Optional[str] = None):
        self.indexer = LocalIndexer()
        self.cache = VLMCache()
        self.vlm_client = GroqVLMClient(api_key=api_key)
        self.processor = VideoProcessor()

    def ensure_indexed(self, video_path: str, verbose_callback: Optional[Callable[[str], None]] = None) -> str:
        """Ensure video is ingested and local frame vectors & tags are stored in SQLite."""
        path = Path(video_path)
        video_id = path.stem

        # Check if already indexed
        with self.indexer._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT video_id FROM videos WHERE video_id = ?", (video_id,))
            if cursor.fetchone():
                if verbose_callback:
                    verbose_callback(f"Video '{video_id}' is already locally indexed.")
                return video_id

        if verbose_callback:
            verbose_callback(f"Indexing video '{video_path}'... detecting scene shots and keyframes.")

        shots = self.processor.process_video(video_path)
        self.indexer.index_video(video_path, shots)

        if verbose_callback:
            verbose_callback(f"Successfully indexed {len(shots)} shots for '{video_id}'.")

        return video_id

    def answer_query(
        self,
        video_path: str,
        query: str,
        verbose_callback: Optional[Callable[[str], None]] = None
    ) -> Dict[str, Any]:
        """Process user query using two-tier vector search and rate-limited Groq VLM priority queue."""
        video_id = self.ensure_indexed(video_path, verbose_callback=verbose_callback)

        # 1. Search existing VLM cache first (Fast Path)
        cached_matches = self.cache.search_observations(video_id, query, top_k=2)
        if cached_matches:
            top_obs, sim_score = cached_matches[0]
            if sim_score >= 0.45:
                if verbose_callback:
                    verbose_callback(f"Cache Hit! Found matching VLM observation (similarity: {sim_score:.2f}).")
                
                # Format context text from cached observation
                events_summary = []
                for ev in top_obs["raw_json"].get("events", []):
                    events_summary.append(
                        f"[{ev.get('start_time')}-{ev.get('end_time')}] {ev.get('description')} "
                        f"(Objects: {', '.join(ev.get('visible_objects', []))})"
                    )
                context_facts = "\n".join(events_summary)

                answer = self.vlm_client.generate_text_answer(query, context_facts)
                return {
                    "answer": answer,
                    "source": "vlm_cache",
                    "shot_id": top_obs["shot_id"],
                    "similarity": sim_score,
                    "observations": top_obs["raw_json"]
                }

        # 2. Perform CLIP visual frame vector search to find candidate shots
        if verbose_callback:
            verbose_callback(f"Searching local CLIP frame embeddings for query: '{query}'...")

        candidate_shots = self.indexer.search_shots(video_id, query, top_k=2)
        if not candidate_shots:
            return {
                "answer": "No matching scenes or keyframes found in the video.",
                "source": "none"
            }

        top_shot, clip_score = candidate_shots[0]
        shot_id = top_shot["shot_id"]
        if verbose_callback:
            verbose_callback(
                f"Top candidate shot selected: {shot_id} [{top_shot['start_ts']}-{top_shot['end_ts']}] "
                f"(CLIP visual score: {clip_score:.2f})."
            )

        # Check if this shot has a cached observation
        existing_obs = self.cache.get_observation(shot_id)
        if existing_obs:
            if verbose_callback:
                verbose_callback(f"Found cached VLM analysis for shot {shot_id}.")
            raw_json = existing_obs["raw_json"]
        else:
            # 3. Analyze candidate storyboard with Groq VLM (Rate Limited 60s)
            if verbose_callback:
                verbose_callback(f"Sending storyboard for shot {shot_id} to Groq VLM qwen/qwen3.6-27b...")
            
            raw_json = self.vlm_client.analyze_storyboard(
                storyboard_b64=top_shot["storyboard_b64"],
                shot_info=top_shot,
                verbose_callback=verbose_callback
            )

            # Build readable summary text for embedding
            summary_lines = []
            for ev in raw_json.get("events", []):
                summary_lines.append(f"{ev.get('start_time')}-{ev.get('end_time')}: {ev.get('description')}")
            summary_text = " ".join(summary_lines)

            # Cache the observation
            self.cache.save_observation(
                shot_id=shot_id,
                video_id=video_id,
                raw_json_data=raw_json,
                summary_text=summary_text
            )

        # Build context facts and generate grounded answer
        facts = []
        for ev in raw_json.get("events", []):
            facts.append(f"[{ev.get('start_time')}-{ev.get('end_time')}] {ev.get('description')}")
        context_str = "\n".join(facts)

        answer = self.vlm_client.generate_text_answer(query, context_str)
        return {
            "answer": answer,
            "source": "groq_vlm",
            "shot_id": shot_id,
            "clip_score": clip_score,
            "observations": raw_json
        }

    def summarize_video(self, video_path: str, verbose_callback: Optional[Callable[[str], None]] = None) -> str:
        """Generate a complete video summary from all indexed/cached observations."""
        video_id = self.ensure_indexed(video_path, verbose_callback=verbose_callback)

        with self.cache._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM vlm_observations WHERE video_id = ? ORDER BY shot_id ASC", (video_id,))
            rows = cursor.fetchall()

        if not rows:
            return f"No VLM observations cached for video '{video_id}' yet. Run some queries first or index offline."

        all_facts = []
        for row in rows:
            obs = json.loads(row["raw_json"])
            for ev in obs.get("events", []):
                all_facts.append(f"[{ev.get('start_time')}-{ev.get('end_time')}] {ev.get('description')}")

        context = "\n".join(all_facts)
        return self.vlm_client.generate_text_answer("Summarize the main sequence of events in this video.", context)
