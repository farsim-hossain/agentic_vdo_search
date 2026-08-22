import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple
from src.video.processor import VideoProcessor
from src.indexing.local_indexer import LocalIndexer, parse_query_timestamp_range
from src.vlm.client import GroqVLMClient
from src.vlm.cache import VLMCache
from src.llamaindex.index_builder import LlamaVideoIndexBuilder

class AgenticRouter:
    def __init__(self, api_key: Optional[str] = None):
        self.indexer = LocalIndexer()
        self.cache = VLMCache()
        self.vlm_client = GroqVLMClient(api_key=api_key)
        self.processor = VideoProcessor()
        self.llama_builder = LlamaVideoIndexBuilder(self.indexer)

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
        """Process user query using LlamaIndex node retrieval, two-tier vector search, and rate-limited Groq VLM."""
        video_id = self.ensure_indexed(video_path, verbose_callback=verbose_callback)

        # Direct Routing: Handle global summary intent
        query_low = query.lower().strip()
        if any(k in query_low for k in ["summarize", "summary", "overall narrative", "tell me everything"]):
            if verbose_callback:
                verbose_callback("Summary intent detected. Aggregating video observations...")
            summary_ans = self.summarize_video(video_path, verbose_callback=verbose_callback)
            return {
                "answer": summary_ans,
                "source": "video_summary"
            }

        # 1. Search existing VLM cache first (Fast Path)
        cached_matches = self.cache.search_observations(video_id, query, top_k=3)
        if cached_matches:
            top_obs, sim_score = cached_matches[0]
            if sim_score >= 0.45:
                if verbose_callback:
                    verbose_callback(f"Cache Hit! Found matching VLM observation (similarity: {sim_score:.2f}).")
                
                # Combine facts across top cached matches
                events_summary = []
                combined_events = []
                for obs_item, s_score in cached_matches:
                    for ev in obs_item["raw_json"].get("events", []):
                        combined_events.append(ev)
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
                    "observations": {"events": combined_events}
                }

        # 2. Perform LlamaIndex ImageNode retrieval & CLIP visual vector search
        if verbose_callback:
            verbose_callback(f"Searching LlamaIndex ImageNodes and CLIP frame embeddings for query: '{query}'...")

        v_duration = 0.0
        with self.indexer._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT duration_sec FROM videos WHERE video_id = ?", (video_id,))
            v_row = cursor.fetchone()
            if v_row and v_row["duration_sec"]:
                v_duration = v_row["duration_sec"]

        ts_range = parse_query_timestamp_range(query, video_duration=v_duration)
        candidate_shots = self.indexer.search_shots(video_id, query, top_k=5)
        
        if not candidate_shots:
            return {
                "answer": "No matching scenes or keyframes found in the video.",
                "source": "none"
            }

        # Filter ONLY shots overlapping target timestamp interval
        if ts_range:
            t_start, t_end = ts_range
            overlapping_candidates = []
            for shot_dict, score in candidate_shots:
                s_start = shot_dict["start_sec"]
                s_end = shot_dict["end_sec"]
                if (s_start <= t_end and s_end >= t_start) or (t_start == t_end and s_start <= t_start <= s_end):
                    overlapping_candidates.append((shot_dict, score))
            if overlapping_candidates:
                candidate_shots = overlapping_candidates

        if verbose_callback:
            verbose_callback(
                f"LlamaIndex selected {len(candidate_shots)} candidate ImageNode(s) covering target query interval."
            )

        # Collect facts across candidate shots, making AT MOST 1 new VLM call per query
        combined_facts = []
        combined_events = []
        primary_shot_id = candidate_shots[0][0]["shot_id"]
        vlm_calls_made = 0

        for shot_dict, score in candidate_shots:
            s_id = shot_dict["shot_id"]
            existing_obs = self.cache.get_observation(s_id)
            if existing_obs:
                if verbose_callback:
                    verbose_callback(f"Found cached VLM analysis for ImageNode {s_id} [{shot_dict['start_ts']}-{shot_dict['end_ts']}].")
                raw_json = existing_obs["raw_json"]
            else:
                if vlm_calls_made >= 1:
                    continue

                if verbose_callback:
                    verbose_callback(f"Sending ImageNode storyboard for shot {s_id} [{shot_dict['start_ts']}-{shot_dict['end_ts']}] to Groq VLM...")
                raw_json = self.vlm_client.analyze_storyboard(
                    storyboard_b64=shot_dict["storyboard_b64"],
                    shot_info=shot_dict,
                    verbose_callback=verbose_callback
                )
                vlm_calls_made += 1
                facts_text_list = []
                for ev in raw_json.get("events", []):
                    line = f"{ev.get('start_time')}-{ev.get('end_time')}: {ev.get('description', '')}"
                    if ev.get("physical_interactions"):
                        line += f" (Interactions: {', '.join(ev.get('physical_interactions'))})"
                    if ev.get("suspicion_rating") and ev.get("suspicion_rating") != "NONE":
                        line += f" [Suspicion: {ev.get('suspicion_rating')} - {ev.get('suspicion_reason', '')}]"
                    if ev.get("recommendation"):
                        line += f" [{ev.get('recommendation')}]"
                    facts_text_list.append(line)

                self.cache.save_observation(
                    shot_id=s_id,
                    video_id=video_id,
                    raw_json_data=raw_json,
                    summary_text=" ".join(facts_text_list)
                )

            for ev in raw_json.get("events", []):
                combined_events.append(ev)
                fact_line = f"[{ev.get('start_time')}-{ev.get('end_time')}] Description: {ev.get('description', '')}"
                if ev.get("physical_interactions"):
                    fact_line += f" | Interactions: {', '.join(ev.get('physical_interactions'))}"
                if ev.get("object_state_changes"):
                    fact_line += f" | State Changes: {', '.join(ev.get('object_state_changes'))}"
                if ev.get("tempo"):
                    fact_line += f" | Tempo: {ev.get('tempo')}"
                if ev.get("suspicion_rating") and ev.get("suspicion_rating") != "NONE":
                    fact_line += f" | Suspicion Rating: {ev.get('suspicion_rating')} ({ev.get('suspicion_reason', '')})"
                if ev.get("recommendation"):
                    fact_line += f" | Recommendation: {ev.get('recommendation')}"
                combined_facts.append(fact_line)

        context_str = "\n".join(combined_facts)
        answer = self.vlm_client.generate_text_answer(query, context_str)

        return {
            "answer": answer,
            "source": "groq_vlm",
            "shot_id": primary_shot_id,
            "clip_score": candidate_shots[0][1],
            "observations": {"events": combined_events}
        }

    def summarize_video(self, video_path: str, verbose_callback: Optional[Callable[[str], None]] = None) -> str:
        """Generate a complete video summary from all indexed/cached observations."""
        video_id = self.ensure_indexed(video_path, verbose_callback=verbose_callback)

        with self.cache._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM vlm_observations WHERE video_id = ? ORDER BY shot_id ASC", (video_id,))
            rows = cursor.fetchall()

        if not rows:
            with self.indexer._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM shots WHERE video_id = ? ORDER BY shot_index ASC LIMIT 2", (video_id,))
                shot_rows = cursor.fetchall()

            for s_row in shot_rows:
                s_id = s_row["shot_id"]
                if s_row["storyboard_b64"]:
                    shot_info = {
                        "start_ts": s_row["start_ts"],
                        "end_ts": s_row["end_ts"],
                        "tags": json.loads(s_row["tags_json"] or "[]")
                    }
                    raw_json = self.vlm_client.analyze_storyboard(
                        storyboard_b64=s_row["storyboard_b64"],
                        shot_info=shot_info,
                        verbose_callback=verbose_callback
                    )
                    summary_lines = [f"{ev.get('start_time')}-{ev.get('end_time')}: {ev.get('description')}" for ev in raw_json.get("events", [])]
                    self.cache.save_observation(
                        shot_id=s_id,
                        video_id=video_id,
                        raw_json_data=raw_json,
                        summary_text=" ".join(summary_lines)
                    )

            with self.cache._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM vlm_observations WHERE video_id = ? ORDER BY shot_id ASC", (video_id,))
                rows = cursor.fetchall()

        all_facts = []
        for row in rows:
            obs = json.loads(row["raw_json"])
            for ev in obs.get("events", []):
                all_facts.append(f"[{ev.get('start_time')}-{ev.get('end_time')}] {ev.get('description')}")

        context = "\n".join(all_facts)
        return self.vlm_client.generate_text_answer("Summarize the main sequence of events in this video.", context)
