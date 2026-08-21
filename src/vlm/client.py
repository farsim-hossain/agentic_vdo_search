import os
import json
from typing import Dict, Any, Optional, Callable
from groq import Groq
from src.config import settings
from src.vlm.rate_limiter import VLMRateLimiter

class GroqVLMClient:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        key = api_key or settings.groq_api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY is not set. Please set it in .env file.")
        self.client = Groq(api_key=key)
        self.vlm_model = model or settings.groq_vlm_model
        self.text_model = settings.groq_text_model

    def analyze_storyboard(
        self,
        storyboard_b64: str,
        shot_info: Dict[str, Any],
        verbose_callback: Optional[Callable[[str], None]] = None
    ) -> Dict[str, Any]:
        """Analyze timestamped storyboard contact sheet image using Groq VLM qwen/qwen3.6-27b with JSON mode."""
        # Enforce rate limiter 60-second cooldown
        rate_limiter = VLMRateLimiter.get_instance()
        rate_limiter.wait_if_needed(verbose_callback=verbose_callback)

        prompt_text = (
            f"Analyze this video storyboard contact sheet representing a shot from {shot_info.get('start_ts', '00:00:00')} "
            f"to {shot_info.get('end_ts', '00:00:00')}.\n"
            "Each frame has an embedded timestamp badge in the corner: [HH:MM:SS].\n"
            "Return valid JSON only with a top-level key 'events' containing a list of observed visual events.\n"
            "Each event object must include:\n"
            "  - start_time: string timestamp\n"
            "  - end_time: string timestamp\n"
            "  - description: detailed visual description of what occurs\n"
            "  - visible_objects: list of strings (objects, people, text visible)\n"
            "  - action: string key action\n"
            "  - confidence: float score between 0.0 and 1.0"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.vlm_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": storyboard_b64
                                },
                            },
                        ],
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_completion_tokens=1024,
            )

            content = response.choices[0].message.content
            parsed = json.loads(content)
            return parsed
        except Exception as e:
            # Fallback structure on error
            return {
                "events": [
                    {
                        "start_time": shot_info.get("start_ts", "00:00:00"),
                        "end_time": shot_info.get("end_ts", "00:00:00"),
                        "description": f"Video shot from {shot_info.get('start_ts')} to {shot_info.get('end_ts')}.",
                        "visible_objects": shot_info.get("tags", []),
                        "action": "shot scene",
                        "confidence": 0.5,
                        "error": str(e)
                    }
                ]
            }

    def generate_text_answer(self, query: str, context_facts: str) -> str:
        """Use Groq text model (llama-3.3-70b-versatile) for instant synthesis based on cached VLM visual facts."""
        prompt = (
            "You are an intelligent video analytics assistant answering questions based strictly on visual observations.\n\n"
            f"Context Observations:\n{context_facts}\n\n"
            f"User Question: {query}\n\n"
            "Provide a clear, grounded answer citing specific timestamps [HH:MM:SS]."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.text_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_completion_tokens=512,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Answer generated from observations:\n{context_facts}\n(Note: Text model generation error: {e})"
