import os
import json
import re
from typing import Dict, Any, Optional, Callable
from groq import Groq
from src.config import settings
from src.vlm.rate_limiter import VLMRateLimiter

def clean_thinking_trace(text: str) -> str:
    """Strip LLM internal reasoning/thinking traces (<think>...</think>, 'Here's a thinking process:...', or '*Self-Correction*:...')."""
    if not text:
        return ""

    # 1. Strip <think>...</think> tags
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # 2. Search for Final Answer or Final Output markers from the bottom up
    matches = list(re.finditer(r'(?:Final Answer|Final Response|Final Output Construction|Final Output|Final decision|Revised Final|Final Polish):\s*(.*)', text, flags=re.IGNORECASE))
    if matches:
        last_match = matches[-1].group(1).strip()
        cleaned = re.sub(r'\*[^*]+\*.*', '', last_match, flags=re.DOTALL).strip()
        if cleaned:
            return cleaned
        return last_match.strip()

    # 3. Line-by-line filtering for internal monologue markers
    lines = text.split("\n")
    valid_lines = []
    for line in lines:
        l = line.strip().lower()
        if any(l.startswith(k) for k in [
            '*correction*', '*wait*', '*self-correction*', '*revised draft*', '*one more check*',
            '*one last thought*', '*actually*', 'let\'s produce', 'let\'s stick', 'okay.', 'final decision:',
            'thinking process', 'analyze user input', 'identify key constraints', 'formulate response strategy', 'drafting the response', 'refine and format'
        ]):
            continue
        if any(k in l for k in ['thinking process', 'analyze user input', 'formulate response strategy', 'self-correction on']):
            continue
        valid_lines.append(line)

    res = "\n".join(valid_lines).strip()
    return res if res else text.strip()

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
            "Each frame has an embedded timestamp badge: [HH:MM:SS].\n"
            "CRITICAL OBJECTIVITY & TEMPORAL ANOMALY CONSTRAINTS:\n"
            "1. Do NOT assume any person is a security guard, police officer, or car owner based on uniform or clothing.\n"
            "2. Inspect human hands and car/object components frame by frame: Are hands touching, pulling, unscrewing, or detaching any part?\n"
            "3. Calculate DWELL TIME: How many seconds did the subject remain next to the car/object? Is the pace unusually brief, rapid, or abrupt?\n"
            "4. FLAG SUSPICIOUS BEHAVIOR: Evaluate suspicion rating (HIGH, MEDIUM, LOW, NONE) and output a proactive Security Advisory note.\n"
            "Return valid JSON only with a top-level key 'events' containing a list of event objects.\n"
            "Each event object MUST include:\n"
            "  - start_time: string timestamp [HH:MM:SS]\n"
            "  - end_time: string timestamp [HH:MM:SS]\n"
            "  - dwell_time_sec: float seconds spent near vehicle or object\n"
            "  - description: detailed grounded visual description\n"
            "  - visible_objects: list of strings (subjects, people, clothing colors, vehicles, accessories)\n"
            "  - physical_interactions: list of strings (hand-to-object contact, pulling, detaching, or walking together)\n"
            "  - object_state_changes: list of strings (part intact vs missing/detached, or 'None')\n"
            "  - tempo: string entry/exit pace (e.g. rapid 3-second lean and departure)\n"
            "  - suspicion_rating: string (HIGH, MEDIUM, LOW, NONE)\n"
            "  - suspicion_reason: why the short dwell time or hasty pace is suspicious\n"
            "  - recommendation: string security advice note (e.g. '⚠️ You should take a look at [00:00:15 - 00:00:19] due to a suspicious 3-second interaction near an unattended vehicle.')\n"
            "  - activity_classification: string event category (e.g. Theft/Vehicle Tampering, Casual Walk, Conversation, Delivery, Shopping)\n"
            "  - action: string key action summary\n"
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
            "Include any proactive Security Advisory notes ('⚠️ Recommendation: You should take a look at [HH:MM:SS]...') whenever a suspicious 2-3 second dwell time or hasty approach/departure is present in the observations.\n"
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
