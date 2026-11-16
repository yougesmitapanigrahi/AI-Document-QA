import unittest

from src.query_router import retrieve_chunks_for_analysis


class QueryRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            {"page_number": 1, "text": "first page"},
            {"page_number": 2, "text": "second page"},
            {"page_number": 3, "text": "third page"},
        ]

    def test_explicit_page_lookup_filters_chunks_without_semantic_search(self) -> None:
        def unexpected_search() -> list[dict[str, str | int]]:
            raise AssertionError("Page lookup must not invoke semantic search.")

        retrieved = retrieve_chunks_for_analysis(
            {"category": "page_lookup", "page_numbers": [3, 1]},
            self.chunks,
            unexpected_search,
        )

        self.assertEqual(
            retrieved,
            [
                {"page_number": 1, "text": "first page"},
                {"page_number": 3, "text": "third page"},
            ],
        )

    def test_page_lookup_without_page_numbers_uses_semantic_search(self) -> None:
        expected = [{"page_number": 2, "text": "semantic result"}]
        search_calls = 0

        def semantic_search() -> list[dict[str, str | int]]:
            nonlocal search_calls
            search_calls += 1
            return expected

        retrieved = retrieve_chunks_for_analysis(
            {"category": "page_lookup", "page_numbers": []},
            self.chunks,
            semantic_search,
        )

        self.assertEqual(retrieved, expected)
        self.assertEqual(search_calls, 1)

    def test_page_count_does_not_invoke_semantic_search(self) -> None:
        def unexpected_search() -> list[dict[str, str | int]]:
            raise AssertionError("Page count must bypass retrieval.")

        self.assertEqual(
            retrieve_chunks_for_analysis(
                {"category": "page_count", "page_numbers": []},
                self.chunks,
                unexpected_search,
            ),
            [],
        )

    def test_other_categories_fall_back_to_semantic_search(self) -> None:
        for category in (
            "factual",
            "explanation",
            "general_question",
            "summary",
            "comparison",
        ):
            with self.subTest(category=category):
                result = retrieve_chunks_for_analysis(
                    {"category": category, "page_numbers": []},
                    self.chunks,
                    lambda: [self.chunks[1]],
                )
                self.assertEqual(result, [self.chunks[1]])


if __name__ == "__main__":
    unittest.main()
