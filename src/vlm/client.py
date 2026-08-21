import os
import json
import re
from typing import Dict, Any, Optional, Callable
from groq import Groq
from src.config import settings
from src.vlm.rate_limiter import VLMRateLimiter

def clean_thinking_trace(text: str) -> str:
    """Strip LLM internal reasoning/thinking traces (<think>...</think> or 'Here's a thinking process:...')."""
    if not text:
        return ""
    
    # Strip <think>...</think> tags
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # Strip "Here's a thinking process: ... Draft: ... Final Polish:"
    if "thinking process" in text.lower() or "analyze user input" in text.lower():
        markers = [
            r'Final Polish:\s*(.*)',
            r'Final Response:\s*(.*)',
            r'Response:\s*(.*)',
            r'Draft:\s*(.*)',
        ]
        for marker in markers:
            match = re.search(marker, text, flags=re.DOTALL | re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                if candidate and not ("thinking process" in candidate.lower() or "analyze user input" in candidate.lower()):
                    return candidate

        # Fallback line filtering
        lines = text.split("\n")
        final_lines = []
        in_thinking = False
        for line in lines:
            low = line.lower()
            if any(k in low for k in ["thinking process", "analyze user input", "identify key constraints", "formulate response strategy", "drafting the response", "refine and format"]):
                in_thinking = True
                continue
            if any(k in low for k in ["final polish:", "final response:", "based on the visual observations"]):
                in_thinking = False
                if "final polish:" in low or "final response:" in low:
                    line = re.sub(r'^(final polish|final response):\s*', '', line, flags=re.IGNORECASE)
            if not in_thinking:
                final_lines.append(line)
        cleaned = "\n".join(final_lines).strip()
        if cleaned:
            return cleaned

    return text.strip()

class GroqVLMClient:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        key = api_key or settings.groq_api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY is not set. Please set it in .env file.")
        self.client = Groq(api_key=key)
        self.vlm_model = model or settings.groq_vlm_model
        self.text_model = settings.groq_text_model or self.vlm_model

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
                max_completion_tokens=4096,
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
        """Use Groq text/vision model (qwen/qwen3.6-27b) for text reasoning based on visual facts."""
        system_instruction = (
            "You are a video analytics assistant. Provide ONLY a direct, grounded answer for the user citing timestamps [HH:MM:SS].\n"
            "CRITICAL: Do NOT output any internal thinking steps, reasoning process, or draft notes. Start directly with the answer."
        )

        prompt = (
            f"Context Observations:\n{context_facts}\n\n"
            f"User Question: {query}\n\n"
            "Answer directly formatted like: From [HH:MM:SS] to [HH:MM:SS], [description of visual actions]..."
        )

        models_to_try = [self.text_model, "qwen/qwen3.6-27b"]

        for m in models_to_try:
            try:
                response = self.client.chat.completions.create(
                    model=m,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_completion_tokens=4096,
                )
                raw_answer = response.choices[0].message.content.strip()
                cleaned_answer = clean_thinking_trace(raw_answer)
                return cleaned_answer if cleaned_answer else raw_answer
            except Exception:
                continue

        return clean_thinking_trace(f"Observations:\n{context_facts}")
