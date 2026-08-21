import unittest
from src.vlm.client import clean_thinking_trace

class TestThinkingCleaner(unittest.TestCase):
    def test_strip_thinking_process(self):
        raw_text = (
            "Here's a thinking process:\n"
            "Analyze User Input:\n"
            "Role: Intelligent video analytics assistant.\n"
            "Draft:\n"
            "Based on observations...\n"
            "Final Polish:\n"
            "From [00:00:16] to [00:00:25], a black sedan is parked in the alleyway."
        )
        cleaned = clean_thinking_trace(raw_text)
        self.assertNotIn("thinking process", cleaned.lower())
        self.assertNotIn("final polish:", cleaned.lower())
        self.assertTrue(cleaned.startswith("From [00:00:16] to [00:00:25]"))

    def test_strip_think_tags(self):
        raw_text = "<think>Internal reasoning step</think>From [00:00:05] to [00:00:10], a motorcycle is visible."
        cleaned = clean_thinking_trace(raw_text)
        self.assertEqual(cleaned, "From [00:00:05] to [00:00:10], a motorcycle is visible.")

if __name__ == "__main__":
    unittest.main()
