import unittest

from src.question_analyzer import analyze_question, get_invalid_page_references


class QuestionAnalyzerTests(unittest.TestCase):
    def test_page_count_phrasings_use_direct_metadata_strategy(self) -> None:
        questions = (
            "How many pages does this PDF have?",
            "How many pages are in this document?",
            "What is the total page count?",
            "Tell me the number of pages.",
        )

        for question in questions:
            with self.subTest(question=question):
                result = analyze_question(question, total_pages=24)
                self.assertEqual(result["category"], "page_count")
                self.assertEqual(result["strategy"], "direct_metadata")
                self.assertEqual(result["page_numbers"], [])

    def test_comparison_takes_priority_and_expands_page_ranges(self) -> None:
        result = analyze_question("Compare page 2 with pages 4-6 and 2.", total_pages=5)

        self.assertEqual(result["category"], "comparison")
        self.assertEqual(result["strategy"], "comparative_retrieval")
        self.assertEqual(result["page_numbers"], [2, 4, 5])

    def test_page_references_are_deduplicated_and_validated(self) -> None:
        result = analyze_question("What is on pages 2-4, 3 and 7?", total_pages=4)

        self.assertEqual(result["category"], "page_lookup")
        self.assertEqual(result["strategy"], "filter_pages_then_answer")
        self.assertEqual(result["page_numbers"], [2, 3, 4])

    def test_zero_and_negative_page_references_are_rejected(self) -> None:
        for question in ("What is on page 0?", "What is on page -2?", "What is on pages 0-3?"):
            with self.subTest(question=question):
                self.assertEqual(analyze_question(question)["page_numbers"], [])

        mixed_result = analyze_question("What is on pages -1 and 3?")
        self.assertEqual(mixed_result["page_numbers"], [3])

    def test_total_pages_must_be_a_positive_non_boolean_integer(self) -> None:
        invalid_values = (0, -1, True, False, 2.5, "5", [], {})
        for total_pages in invalid_values:
            with self.subTest(total_pages=total_pages):
                with self.assertRaises(ValueError):
                    analyze_question("What is on page 1?", total_pages=total_pages)

    def test_page_numbers_are_bounded_by_total_pages(self) -> None:
        result = analyze_question("What is on pages 1, 5, 6 and 0?", total_pages=5)
        self.assertEqual(result["page_numbers"], [1, 5])

    def test_explicit_invalid_page_references_can_be_distinguished(self) -> None:
        self.assertEqual(
            get_invalid_page_references("What is on pages 2 and 9?", 5),
            [9],
        )
        self.assertEqual(get_invalid_page_references("What is on page 0?", 5), [0])
        self.assertEqual(
            get_invalid_page_references("What is on pages 2-8?", 5),
            [8],
        )
        self.assertEqual(get_invalid_page_references("What is on page 3?", 5), [])
        self.assertIsNone(
            get_invalid_page_references("What pages discuss risks?", 5)
        )

    def test_descending_ranges_preserve_order_and_validate_bounds(self) -> None:
        result = analyze_question("What is on pages 8-2?", total_pages=5)
        self.assertEqual(result["page_numbers"], [5, 4, 3, 2])

    def test_extremely_large_ranges_are_ignored(self) -> None:
        result = analyze_question("What is on pages 1-1000000000?")
        self.assertEqual(result["page_numbers"], [])

    def test_question_categories_and_strategies(self) -> None:
        cases = (
            ("Who wrote the report?", "factual", "semantic_search"),
            ("Why did the results change?", "explanation", "semantic_search"),
            ("Summarize the main findings.", "summary", "document_summary"),
            ("Please help me.", "general_question", "semantic_search"),
        )

        for question, category, strategy in cases:
            with self.subTest(question=question):
                result = analyze_question(question)
                self.assertEqual(result["category"], category)
                self.assertEqual(result["strategy"], strategy)

    def test_keywords_topics_and_empty_question(self) -> None:
        result = analyze_question("Explain the climate change impacts.")
        self.assertEqual(result["keywords"], ["climate", "change", "impacts"])
        self.assertEqual(result["topics"], ["climate change", "change impacts"])

        empty_result = analyze_question("  ?!  ")
        self.assertEqual(empty_result["category"], "general_question")
        self.assertEqual(empty_result["page_numbers"], [])
        self.assertEqual(empty_result["keywords"], [])
        self.assertEqual(empty_result["topics"], [])
        self.assertEqual(empty_result["strategy"], "semantic_search")


if __name__ == "__main__":
    unittest.main()
