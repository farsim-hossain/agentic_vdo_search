import unittest
import csv
import io
from src.agent.router import AgenticRouter

class TestFullVideoLog(unittest.TestCase):
    def setUp(self):
        self.router = AgenticRouter()
        self.test_video = "input_vdo/Stealing095_x264.mp4"

    def test_generate_full_video_log(self):
        res = self.router.generate_full_video_log(self.test_video, mode="zero_llm")
        self.assertIn("summary", res)
        self.assertIn("events_log", res)
        self.assertIn("csv_payload", res)

        events_log = res["events_log"]
        self.assertGreater(len(events_log), 0)

        first_row = events_log[0]
        expected_keys = [
            "Shot #", "Start Time", "End Time", "Duration (s)",
            "Detected Objects", "Time Talks (Pace)", "Visual Activity", "Security Recommendation"
        ]
        for key in expected_keys:
            self.assertIn(key, first_row)

        # Validate CSV parsing
        csv_file = io.StringIO(res["csv_payload"])
        reader = csv.DictReader(csv_file)
        csv_rows = list(reader)
        self.assertEqual(len(csv_rows), len(events_log))

if __name__ == "__main__":
    unittest.main()
