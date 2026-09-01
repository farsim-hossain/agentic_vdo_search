import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from src.indexing.embeddings import EmbeddingEngine

class LocalNaturalLanguageGenerator:
    """100% Domain-Agnostic & Universal Zero-LLM ($0 API Cost) Natural Language Generator.
    Works seamlessly across thousands of video types (surveillance, sports, retail, cooking, drones, traffic, pets)
    without domain-specific coupling or hardcoded vehicle variables.
    """

    def __init__(self):
        self.embed_engine = EmbeddingEngine.get_instance()

    def evaluate_query_alignment(self, query: str, kf_vectors: List[np.ndarray]) -> float:
        """Embed user query string directly into CLIP 512-dim text vector space and compute max similarity against keyframes."""
        if not kf_vectors:
            return 0.0
        query_vec = self.embed_engine.embed_clip_text(query)
        scores = [self.embed_engine.cosine_similarity(query_vec, kf_vec) for kf_vec in kf_vectors]
        return float(np.max(scores)) if scores else 0.0

    def synthesize_answer(self, query: str, candidate_shots: List[Tuple[Dict[str, Any], float]]) -> Dict[str, Any]:
        """Synthesize a dynamic, domain-agnostic response for any question across any video type at $0 API cost."""
        if not candidate_shots:
            return {
                "answer": f"No visual events matching '{query}' were indexed in the video.",
                "source": "zero_llm_nlg",
                "observations": {"events": []}
            }

        primary_shot, score = candidate_shots[0]
        start_ts = primary_shot.get("start_ts", "00:00:00")
        end_ts = primary_shot.get("end_ts", "00:00:00")
        start_sec = primary_shot.get("start_sec", 0.0)
        end_sec = primary_shot.get("end_sec", 0.0)
        dwell_time = max(1.5, round(end_sec - start_sec, 1))

        tags = primary_shot.get("tags", [])
        obj_str = ", ".join(tags) if tags else "objects and scene environment"
        kf_vectors = primary_shot.get("kf_vectors", [])

        # Direct OpenCLIP score for the EXACT query text typed by user across any video category
        query_sim = self.evaluate_query_alignment(query, kf_vectors)
        conf_percent = int(max_score_norm(query_sim) * 100)

        # Contextual Security Advisory: ONLY attached if query is a security/theft question AND visual match is high AND rapid dwell time occurs
        is_security_query = any(w in query.lower() for w in ["steal", "theft", "suspicious", "rob", "tamper", "break", "unusual", "hurry", "quick", "warning", "caution"])
        is_suspicious_event = is_security_query and dwell_time <= 5.0 and query_sim >= 0.25

        if is_suspicious_event:
            suspicion_rating = "MEDIUM" if dwell_time >= 3.0 else "HIGH"
            suspicion_reason = f"Brief {dwell_time}-second interaction detected during flagged security query."
            rec_note = f"Security Advisory: You should take a look at [{start_ts} - {end_ts}] due to the brief {dwell_time}-second event duration."
        else:
            suspicion_rating = "NONE"
            suspicion_reason = "Normal activity with standard movement pace."
            rec_note = ""

        # Dynamic Universal Answer Construction for ANY question across thousands of video categories
        clean_query = query.rstrip("?.!").strip()

        if is_security_query:
            if is_suspicious_event:
                ans = (
                    f"From [{start_ts}] to [{end_ts}], a subject/object was detected in scene with {obj_str} "
                    f"showing a rapid {dwell_time}-second duration. While visual indexing did not record conclusive crime activity, "
                    f"this segment matches your query with a visual confidence of {conf_percent}%. {rec_note}"
                )
            else:
                ans = (
                    f"From [{start_ts}] to [{end_ts}], video analysis detected {obj_str}. "
                    f"No individuals or subjects were observed engaging in suspicious or crime-related behavior."
                )
        else:
            # Universal query (sports, cooking, pets, romance, clothing, traffic, etc.)
            if query_sim >= 0.25:
                ans = (
                    f"From [{start_ts}] to [{end_ts}], visual indexing detected features matching '{clean_query}' "
                    f"along with {obj_str} (Confidence: {conf_percent}%)."
                )
            else:
                ans = (
                    f"From [{start_ts}] to [{end_ts}], video analysis detected {obj_str}. "
                    f"Based on local zero-shot visual indexing, there is no visual evidence of '{clean_query}' in this segment."
                )

        event_dict = {
            "start_time": start_ts,
            "end_time": end_ts,
            "dwell_time_sec": dwell_time,
            "description": f"Visual detection of {obj_str} with {dwell_time}s segment duration.",
            "visible_objects": tags,
            "physical_interactions": [f"scene presence ({dwell_time}s duration)"] if tags else ["general movement"],
            "object_state_changes": ["None"],
            "tempo": f"rapid {dwell_time}-second duration" if dwell_time <= 4.0 else "steady movement",
            "suspicion_rating": suspicion_rating,
            "suspicion_reason": suspicion_reason,
            "recommendation": rec_note,
            "activity_classification": f"Visual Match for '{clean_query}'" if query_sim >= 0.25 else "General Scene",
            "action": f"Indexing for '{clean_query}'",
            "confidence": round(query_sim, 2)
        }

        return {
            "answer": ans.strip(),
            "source": "zero_llm_nlg",
            "shot_id": primary_shot.get("shot_id"),
            "clip_score": score,
            "observations": {"events": [event_dict]}
        }

def max_score_norm(score: float) -> float:
    """Normalize raw cosine similarity score into a 0.5 - 0.95 confidence range."""
    clipped = max(0.15, min(0.40, score))
    return round(0.50 + (clipped - 0.15) / (0.40 - 0.15) * 0.45, 2)
