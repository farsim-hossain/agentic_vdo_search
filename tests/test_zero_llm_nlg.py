import unittest
from src.agent.local_nlg import LocalNaturalLanguageGenerator

class TestZeroLLMNLG(unittest.TestCase):
    def setUp(self):
        self.nlg = LocalNaturalLanguageGenerator()

    def test_synthesize_answer_stealing_query(self):
        candidate_shots = [
            (
                {
                    "shot_id": "test_shot_1",
                    "start_ts": "00:00:15",
                    "end_ts": "00:00:19",
                    "start_sec": 15.0,
                    "end_sec": 19.0,
                    "tags": ["person", "car", "motorcycle"],
                    "kf_vectors": []
                },
                0.85
            )
        ]
        res = self.nlg.synthesize_answer("is anyone stealing anything?", candidate_shots)
        self.assertIn("answer", res)
        self.assertIn("00:00:15", res["answer"])
        self.assertEqual(res["source"], "zero_llm_nlg")
        self.assertIn("events", res["observations"])
        self.assertGreater(len(res["observations"]["events"]), 0)

if __name__ == "__main__":
    unittest.main()
