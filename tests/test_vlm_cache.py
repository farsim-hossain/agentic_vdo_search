import unittest
import tempfile
from pathlib import Path
from src.vlm.cache import VLMCache

class TestVLMCache(unittest.TestCase):
    def test_vlm_cache_sqlite(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            db_path = Path(tmp.name)
            cache = VLMCache(db_path=db_path)
            
            raw_json = {
                "events": [
                    {
                        "start_time": "00:00:00",
                        "end_time": "00:00:05",
                        "description": "A red sports car accelerates down a track.",
                        "visible_objects": ["car", "road"],
                        "action": "driving",
                        "confidence": 0.9
                    }
                ]
            }
            
            cache.save_observation(
                shot_id="vid1_shot_0",
                video_id="vid1",
                raw_json_data=raw_json,
                summary_text="00:00:00-00:00:05: A red sports car accelerates down a track."
            )
            
            obs = cache.get_observation("vid1_shot_0")
            self.assertIsNotNone(obs)
            self.assertEqual(obs["shot_id"], "vid1_shot_0")
            self.assertEqual(obs["raw_json"]["events"][0]["visible_objects"], ["car", "road"])

if __name__ == "__main__":
    unittest.main()
