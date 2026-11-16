import unittest

from src.qa_engine import generate_answer, get_page_count_answer


class PageCountAnswerTests(unittest.TestCase):
    def test_page_count_question_variations_use_provided_total(self) -> None:
        questions = (
            "How many pages does this PDF have?",
            "How many pages are in this document?",
            "What is the total page count?",
            "Tell me the number of pages.",
        )
        expected = {
            "answer": "This PDF contains 24 pages.",
            "source_pages": [],
        }

        for question in questions:
            with self.subTest(question=question):
                self.assertEqual(get_page_count_answer(question, 24), expected)

    def test_generate_answer_short_circuits_before_reading_chunks_or_model(self) -> None:
        class ChunksMustNotBeRead:
            def __iter__(self):
                raise AssertionError("Page-count questions must bypass chunks.")

        result = generate_answer(
            "How many pages does this PDF have?",
            ChunksMustNotBeRead(),
            tokenizer=None,
            model=None,
            total_pages=24,
        )

        self.assertEqual(
            result,
            {"answer": "This PDF contains 24 pages.", "source_pages": []},
        )

    def test_normal_questions_do_not_match_page_count_handler(self) -> None:
        self.assertIsNone(
            get_page_count_answer("What are the main findings?", 24)
        )


if __name__ == "__main__":
    unittest.main()
