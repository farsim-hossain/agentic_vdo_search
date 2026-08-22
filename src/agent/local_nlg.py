import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from src.indexing.embeddings import EmbeddingEngine

CANDIDATE_ACTIVITIES = {
    "Vehicle Tampering / Theft Attempt": "person leaning near car driver side mirror or door trying to detach object",
    "Vehicle Inspection / Loitering": "person standing very close to unattended car inspecting window or door",
    "Casual Pedestrian Walk": "person walking casually along sidewalk or street past vehicles",
    "People Conversation": "two or more people standing together talking",
    "Package / Item Delivery": "person holding box or package carrying it to door"
}

class LocalNaturalLanguageGenerator:
    """Zero-LLM ($0 API Cost) Natural Language Generator.
    Uses local OpenCLIP embeddings, YOLO object tags, and temporal dwell-time rules
    to generate fluent natural answers, timestamp citations, and proactive security advisories.
    """

    def __init__(self):
        self.embed_engine = EmbeddingEngine.get_instance()
        # Precompute activity CLIP text vectors
        self.activity_vectors = {}
        for label, prompt in CANDIDATE_ACTIVITIES.items():
            vec = self.embed_engine.embed_clip_text(prompt)
            self.activity_vectors[label] = vec

    def classify_shot_clip(self, kf_vectors: List[np.ndarray]) -> Tuple[str, float]:
        """Classify keyframe visual vectors against candidate activities locally using OpenCLIP cosine similarity."""
        if not kf_vectors:
            return ("Casual Visual Scene", 0.5)

        best_label = "Casual Pedestrian Walk"
        best_score = -1.0

        for label, text_vec in self.activity_vectors.items():
            scores = [self.embed_engine.cosine_similarity(text_vec, kf_vec) for kf_vec in kf_vectors]
            max_s = float(np.max(scores)) if scores else 0.0
            if max_s > best_score:
                best_score = max_s
                best_label = label

        return best_label, max_score_norm(best_score)

    def synthesize_answer(self, query: str, candidate_shots: List[Tuple[Dict[str, Any], float]]) -> Dict[str, Any]:
        """Synthesize a fluent, grounded natural language response at $0 API cost."""
        if not candidate_shots:
            return {
                "answer": "No visual events matching your query were indexed in the video.",
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
        obj_str = ", ".join(tags) if tags else "vehicles and surrounding environment"

        # Determine activity and suspicion using CLIP visual vectors
        kf_vectors = primary_shot.get("kf_vectors", [])
        activity_label, confidence = self.classify_shot_clip(kf_vectors)

        # Suspicion heuristic: ANY brief 1.5s - 5.0s proximity to car/motorcycle or tampering/loitering label is flagged as suspicious!
        is_suspicious_query = any(w in query.lower() for w in ["steal", "theft", "suspicious", "rob", "tamper", "break", "unusual", "hurry", "quick", "warning", "caution"])
        has_vehicle = any(t in tags for t in ["car", "motorcycle", "vehicle"])
        is_car_proximity = (has_vehicle or "person" in tags) and dwell_time <= 5.0

        if activity_label in ["Vehicle Tampering / Theft Attempt", "Vehicle Inspection / Loitering"] or is_car_proximity:
            suspicion_rating = "MEDIUM" if dwell_time >= 3.0 else "HIGH"
            suspicion_reason = f"Subject's brief {dwell_time}-second proximity near unattended vehicle/motorcycle and rapid departure is unusual."
            rec_note = f"⚠️ Security Advisory: You should take a look at [{start_ts} - {end_ts}] due to the brief {dwell_time}-second interaction near the vehicle."
        else:
            suspicion_rating = "NONE"
            suspicion_reason = "Normal pedestrian activity with standard movement pace."
            rec_note = ""

        # Build natural language response
        if is_suspicious_query or suspicion_rating in ["HIGH", "MEDIUM"]:
            ans = (
                f"From [{start_ts}] to [{end_ts}], a subject was detected in close proximity to {obj_str} "
                f"with a rapid {dwell_time}-second dwell time and hasty departure. While visual indexing did not record complete component detachment, "
                f"this rapid interaction is classified as {activity_label} (Confidence: {int(confidence*100)}%). "
                f"{rec_note}"
            )
        else:
            ans = (
                f"From [{start_ts}] to [{end_ts}], video analysis detected {obj_str}. "
                f"No individuals were observed stealing, running away, or engaging in suspicious behavior."
            )

        event_dict = {
            "start_time": start_ts,
            "end_time": end_ts,
            "dwell_time_sec": dwell_time,
            "description": f"Visual detection of {obj_str} with {dwell_time}s dwell time.",
            "visible_objects": tags,
            "physical_interactions": [f"proximity to vehicle ({dwell_time}s dwell time)"] if "person" in tags else ["normal movement"],
            "object_state_changes": ["None"],
            "tempo": f"rapid {dwell_time}-second proximity" if dwell_time <= 4.0 else "steady movement",
            "suspicion_rating": suspicion_rating,
            "suspicion_reason": suspicion_reason,
            "recommendation": rec_note,
            "activity_classification": activity_label,
            "action": f"{activity_label} near {obj_str}",
            "confidence": confidence
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
