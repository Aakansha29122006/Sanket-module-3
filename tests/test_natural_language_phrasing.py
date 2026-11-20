"""
Natural Language Formatting Tests
Validates exact natural language phrase outputs requested by the user.
"""

import unittest
from road_blockage.detector import generate_natural_language_summary


class TestNaturalLanguagePhrasing(unittest.TestCase):
    """Test standard road blockage summary phrasing."""

    def test_clear_road_zero_percent(self):
        summary = generate_natural_language_summary(0.0, [])
        self.assertEqual(summary, "0% road is clear")

    def test_fallen_trees_and_poles_95_percent(self):
        summary = generate_natural_language_summary(95.0, ["fallen_tree", "fallen_pole"])
        self.assertEqual(summary, "95% road is blocked due to fallen trees and poles")

    def test_loose_soil_10_percent(self):
        summary = generate_natural_language_summary(10.0, ["loose_soil"])
        self.assertEqual(summary, "10% road is blocked due to loose soil")

    def test_fallen_rocks_34_percent(self):
        summary = generate_natural_language_summary(34.0, ["fallen_rocks"])
        self.assertEqual(summary, "34% road is blocked due to fallen rocks")

    def test_accident_100_percent(self):
        summary = generate_natural_language_summary(100.0, ["accident"])
        self.assertEqual(summary, "100% road is blocked due to accident")

    def test_three_obstacles_formatting(self):
        summary = generate_natural_language_summary(80.0, ["fallen_tree", "fallen_pole", "fallen_rocks"])
        self.assertEqual(summary, "80% road is blocked due to fallen trees, poles, and fallen rocks")


if __name__ == "__main__":
    unittest.main()
