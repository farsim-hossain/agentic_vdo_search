import time
import threading
from src.config import settings

class VLMRateLimiter:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, interval_seconds: float = None):
        self.interval = interval_seconds if interval_seconds is not None else settings.vlm_min_request_interval
        self.last_request_time = 0.0

    @classmethod
    def get_instance(cls) -> "VLMRateLimiter":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def seconds_until_next_allowed(self) -> float:
        """Returns the number of seconds remaining before a new VLM call is permitted."""
        elapsed = time.time() - self.last_request_time
        remaining = self.interval - elapsed
        return max(0.0, remaining)

    def wait_if_needed(self, verbose_callback=None):
        """Block until the 60-second rate limit cooldown has expired."""
        with self._lock:
            remaining = self.seconds_until_next_allowed()
            if remaining > 0:
                if verbose_callback:
                    verbose_callback(f"Groq VLM rate limit active: waiting {remaining:.1f} seconds before calling API...")
                time.sleep(remaining)
            self.last_request_time = time.time()
