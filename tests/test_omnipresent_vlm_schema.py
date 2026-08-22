import unittest

class TestOmnipresentVLMSchema(unittest.TestCase):
    def test_schema_fields(self):
        sample_event = {
            "start_time": "00:00:15",
            "end_time": "00:00:19",
            "dwell_time_sec": 3.0,
            "description": "A man in a beige uniform approaches a parked black car, leans for 3 seconds, and departs quickly.",
            "visible_objects": ["person", "beige uniform", "black sedan"],
            "physical_interactions": ["hand touching driver side mirror area", "walking away quickly"],
            "object_state_changes": ["side mirror intact"],
            "tempo": "rapid 3-second lean and departure",
            "suspicion_rating": "HIGH",
            "suspicion_reason": "brief 3-second proximity to unattended vehicle with hasty departure",
            "recommendation": "⚠️ You should take a look at [00:00:15 - 00:00:19] due to a suspicious 3-second approach near an unattended vehicle.",
            "activity_classification": "Theft / Vehicle Tampering",
            "action": "detaching side mirror",
            "confidence": 0.95
        }
        self.assertIn("dwell_time_sec", sample_event)
        self.assertIn("tempo", sample_event)
        self.assertIn("suspicion_rating", sample_event)
        self.assertIn("recommendation", sample_event)
        self.assertEqual(sample_event["suspicion_rating"], "HIGH")

if __name__ == "__main__":
    unittest.main()
