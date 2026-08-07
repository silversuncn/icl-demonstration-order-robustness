import unittest
from src.verify_public_results import verify


class PublicResultVerificationTest(unittest.TestCase):
    def test_public_results_match_reported_claims(self):
        self.assertEqual(verify()["status"], "PASS")

    def test_legacy_parser_bookkeeping_is_absent_from_public_aggregate(self):
        result = verify()
        self.assertEqual(result["computed"].get("legacy_parser_fields", ["verification missing"]), [])

    def test_permutation_multiplicity_summary_is_verified(self):
        result = verify()
        self.assertEqual(result["computed"].get("multiplicity_rows", 0), 3)


if __name__ == "__main__":
    unittest.main()
