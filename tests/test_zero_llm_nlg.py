import unittest
from src.agent.local_nlg import LocalNaturalLanguageGenerator

class TestZeroLLMNLG(unittest.TestCase):
    def setUp(self):
        self.nlg = LocalNaturalLanguageGenerator()
        self.candidate_shots = [
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

    def test_romance_query_no_security_warning(self):
        res = self.nlg.synthesize_answer("do you see any romantic thing happening ? any boy and girl kissing", self.candidate_shots)
        self.assertIn("answer", res)
        self.assertIn("no visual evidence", res["answer"].lower())
        self.assertNotIn("Security Advisory", res["answer"])

    def test_clothing_query(self):
        res = self.nlg.synthesize_answer("is anyone wearing a yellow hat", self.candidate_shots)
        self.assertIn("answer", res)
        self.assertIn("no visual evidence", res["answer"].lower())

    def test_stealing_query_with_security_warning(self):
        res = self.nlg.synthesize_answer("is anyone stealing anything?", self.candidate_shots)
        self.assertIn("answer", res)
        self.assertIn("00:00:15", res["answer"])
        self.assertEqual(res["source"], "zero_llm_nlg")

if __name__ == "__main__":
    unittest.main()
