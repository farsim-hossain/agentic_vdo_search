import time
import unittest
from src.vlm.rate_limiter import VLMRateLimiter

class TestRateLimiter(unittest.TestCase):
    def test_rate_limiter(self):
        limiter = VLMRateLimiter(interval_seconds=0.2)
        limiter.last_request_time = time.time()
        
        self.assertGreater(limiter.seconds_until_next_allowed(), 0)
        
        start = time.time()
        limiter.wait_if_needed()
        elapsed = time.time() - start
        
        # Ensure it waited at least 0.15 seconds
        self.assertGreaterEqual(elapsed, 0.15)
        # After updating last_request_time, the next allowed request is interval_seconds in the future
        self.assertAlmostEqual(limiter.seconds_until_next_allowed(), 0.2, delta=0.05)

if __name__ == "__main__":
    unittest.main()
