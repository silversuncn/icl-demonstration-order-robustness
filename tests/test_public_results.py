import unittest
from src.verify_public_results import verify


class PublicResultVerificationTest(unittest.TestCase):
    def test_public_results_match_reported_claims(self):
        self.assertEqual(verify()["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
