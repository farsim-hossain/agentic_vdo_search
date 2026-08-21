import unittest
from PIL import Image
from src.video.processor import VideoProcessor, Keyframe, format_timestamp

class TestVideoProcessor(unittest.TestCase):
    def test_format_timestamp(self):
        self.assertEqual(format_timestamp(0.0), "00:00:00")
        self.assertEqual(format_timestamp(65.0), "00:01:05")
        self.assertEqual(format_timestamp(3661.0), "01:01:01")

    def test_create_storyboard(self):
        img1 = Image.new("RGB", (100, 100), color="red")
        img2 = Image.new("RGB", (100, 100), color="blue")
        
        kf1 = Keyframe(frame_index=0, pts_seconds=0.0, timestamp_str="00:00:00", image=img1)
        kf2 = Keyframe(frame_index=30, pts_seconds=1.0, timestamp_str="00:00:01", image=img2)
        
        storyboard = VideoProcessor.create_storyboard([kf1, kf2], tile_size=(160, 90))
        self.assertIsInstance(storyboard, Image.Image)
        self.assertGreater(storyboard.width, 0)
        self.assertGreater(storyboard.height, 0)

    def test_storyboard_to_base64(self):
        img = Image.new("RGB", (50, 50), color="green")
        b64_str = VideoProcessor.storyboard_to_base64(img)
        self.assertTrue(b64_str.startswith("data:image/jpeg;base64,"))

if __name__ == "__main__":
    unittest.main()
