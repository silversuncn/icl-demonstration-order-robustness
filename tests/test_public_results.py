import unittest
from src.verify_public_results import verify


class PublicResultVerificationTest(unittest.TestCase):
    def test_public_results_match_reported_claims(self):
        result = verify()
        self.assertEqual(result["status"], "PASS", result.get("failures"))

    def test_legacy_parser_bookkeeping_is_absent_from_public_aggregate(self):
        result = verify()
        self.assertEqual(result["computed"].get("legacy_parser_fields", ["verification missing"]), [])

    def test_permutation_multiplicity_summary_is_verified(self):
        result = verify()
        self.assertEqual(result["computed"].get("multiplicity_rows", 0), 3)
        self.assertEqual(result["computed"].get("multiplicity_random_duplicate_strata"), [15, 5, 0])

    def test_v2_public_table_contract(self):
        result = verify()["computed"]
        self.assertIn("seeded_base", result["order_ids"])
        self.assertEqual(result["aggregate_rows_in_public_package"], 1200)
        self.assertEqual(result["evaluation_instances_in_public_package"], 240000)
        self.assertEqual(result["per_sample_rows"], 72000)
        self.assertEqual(result["per_sample_cells"], 360)
        self.assertEqual(result["descriptive_rows"], 6)
        self.assertTrue(result["descriptive_copy_matches"])
        self.assertEqual(result["descriptive_p_value_columns"], [])
        self.assertEqual(result["descriptive_significance_columns"], [])
        self.assertTrue(result["legacy_holm_table_absent"])
        self.assertTrue(result["seeded_base_alias_file_absent"])
        self.assertEqual(result["cluster_test_rows"], 6)
        self.assertEqual(result["cluster_count"], 15)


if __name__ == "__main__":
    unittest.main()
