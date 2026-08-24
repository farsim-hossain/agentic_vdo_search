import unittest
from api.index import app

class TestVercelFastAPIEntrypoint(unittest.TestCase):
    def test_app_export(self):
        self.assertIsNotNone(app)
        self.assertEqual(app.title, "Agentic Video Search & Intelligence API")

if __name__ == "__main__":
    unittest.main()
