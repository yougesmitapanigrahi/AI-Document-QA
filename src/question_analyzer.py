"""Classify questions and extract lightweight routing information."""

import re


_STRATEGIES = {
    "page_count": "direct_metadata",
    "page_lookup": "filter_pages_then_answer",
    "factual": "semantic_search",
    "explanation": "semantic_search",
    "summary": "document_summary",
    "comparison": "comparative_retrieval",
    "general_question": "semantic_search",
}

_STOP_WORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been
    before being below between both but by can could did do does doing down during
    each few for from further had has have having he her here hers herself him
    himself his how i if in into is it its itself just me more most my myself no
    nor not of off on once only or other our ours ourselves out over own same she
    should so some such than that the their theirs them themselves then there
    these they this those through to too under until up very was we were what when
    where which while who whom why will with would you your yours yourself
    yourselves pdf document file page pages number total tell please explain
    explanation elaborate interpret interpretation summarize summarise summary
    overview recap outline highlights compare compares compared comparison
    contrast contrasts difference differences similarity similarities versus vs
    main key
    """.split()
)

_WORD_PATTERN = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)
_MAX_PAGE_RANGE_LENGTH = 1000
_MAX_PAGE_NUMBER_RESULTS = 1000
_PAGE_REFERENCE_PATTERN = re.compile(
    r"\bpages?\s+(?:numbers?\s+)?"
    r"(?P<references>[+-]?\d+(?:\s*(?:-|–|—|to)\s*[+-]?\d+)?"
    r"(?:\s*(?:,\s*(?:and\s+)?|and\s+|&\s*)"
    r"[+-]?\d+(?:\s*(?:-|–|—|to)\s*[+-]?\d+)?"
    r")*)",
    re.IGNORECASE,
)
_PAGE_ITEM_PATTERN = re.compile(
    r"[+-]?\d+(?:\s*(?:-|–|—|to)\s*[+-]?\d+)?",
    re.IGNORECASE,
)
_PAGE_RANGE_PATTERN = re.compile(
    r"([+-]?\d+)(?:\s*(?:-|–|—|to)\s*([+-]?\d+))?",
    re.IGNORECASE,
)

_COMPARISON_PATTERN = re.compile(
    r"\b(?:compare|compares|compared|comparison|contrast|contrasts|"
    r"difference|differences|similarity|similarities|versus|vs)\b"
)
_SUMMARY_PATTERN = re.compile(
    r"\b(?:summarize|summarise|summary|overview|recap|outline|"
    r"highlights|key points|main points|key findings|main findings|"
    r"key takeaways)\b"
)
_EXPLANATION_PATTERN = re.compile(
    r"\b(?:explain|explanation|why|meaning|interpret|interpretation|"
    r"elaborate)\b"
)
_FACTUAL_START_PATTERN = re.compile(
    r"^(?:who|what|when|where|which|whose|whom|how many|how much|"
    r"how often|how long|how far|how old|is|are|was|were|do|does|did|"
    r"has|have|can|could|will|would)\b"
)


def _normalize_question(question: str) -> str:
    """Lowercase a question, replace punctuation with spaces, and collapse whitespace."""
    without_punctuation = re.sub(r"[^\w\s]", " ", question.casefold())
    return " ".join(without_punctuation.split())


def _is_page_count_question(normalized_question: str) -> bool:
    """Recognize requests for the document's overall page count."""
    if re.search(
        r"\b(?:page count|total pages?|total number of pages?)\b",
        normalized_question,
    ):
        return True

    if re.search(
        r"\b(?:tell me|give me|what is|what s|what are)\s+"
        r"(?:the\s+)?number of pages?\b",
        normalized_question,
    ):
        return True

    match = re.search(r"\bhow many pages?\b", normalized_question)
    if match is None:
        return False

    remainder = normalized_question[match.end():].strip()
    if not remainder:
        return True

    return bool(
        re.match(
            r"(?:does|did|do)\b.*\b(?:have|contain)\b|"
            r"are\b.*\b(?:in|there)\b|are there\b",
            remainder,
        )
    )


def _extract_page_numbers(question: str, total_pages: int | None) -> list[int]:
    """Extract valid, unique page references in order with bounded expansion."""
    page_numbers: list[int] = []
    seen: set[int] = set()

    for reference_match in _PAGE_REFERENCE_PATTERN.finditer(question):
        references = reference_match.group("references")
        for item_match in _PAGE_ITEM_PATTERN.finditer(references):
            range_match = _PAGE_RANGE_PATTERN.fullmatch(item_match.group())
            if range_match is None:
                continue

            first = int(range_match.group(1))
            last_text = range_match.group(2)
            last = int(last_text) if last_text is not None else first
            if first < 1 or last < 1:
                continue

            step = 1 if last >= first else -1

            if step > 0:
                if total_pages is not None:
                    last = min(last, total_pages)
                if first > last:
                    continue
                numbers = range(first, last + 1)
            else:
                if total_pages is not None:
                    first = min(first, total_pages)
                if first < last:
                    continue
                numbers = range(first, last - 1, -1)

            if len(numbers) > _MAX_PAGE_RANGE_LENGTH:
                continue

            for page_number in numbers:
                if page_number not in seen:
                    seen.add(page_number)
                    page_numbers.append(page_number)
                    if len(page_numbers) >= _MAX_PAGE_NUMBER_RESULTS:
                        return page_numbers

    return page_numbers


def get_invalid_page_references(
    question: str, total_pages: int
) -> list[int] | None:
    """Return invalid explicitly requested page endpoints, or ``None`` if absent.

    The returned values are unique and ordered as they appear. For a range,
    its endpoints are sufficient to determine whether it exceeds the document.
    """
    if isinstance(total_pages, bool) or not isinstance(total_pages, int) or total_pages <= 0:
        raise ValueError("total_pages must be a positive integer.")
    if not isinstance(question, str):
        return None

    references_found = False
    invalid_pages: list[int] = []
    seen: set[int] = set()

    for reference_match in _PAGE_REFERENCE_PATTERN.finditer(question):
        references_found = True
        references = reference_match.group("references")
        for item_match in _PAGE_ITEM_PATTERN.finditer(references):
            range_match = _PAGE_RANGE_PATTERN.fullmatch(item_match.group())
            if range_match is None:
                continue

            endpoints = (int(range_match.group(1)),)
            if range_match.group(2) is not None:
                endpoints += (int(range_match.group(2)),)

            for page_number in endpoints:
                if not 1 <= page_number <= total_pages and page_number not in seen:
                    seen.add(page_number)
                    invalid_pages.append(page_number)

    if not references_found:
        return None
    return invalid_pages


def _extract_keywords_and_topics(normalized_question: str) -> tuple[list[str], list[str]]:
    """Return unique content words and adjacent content-word phrases."""
    words = _WORD_PATTERN.findall(normalized_question)
    keywords: list[str] = []
    topics: list[str] = []
    current_topic_words: list[str] = []

    def finish_topic_phrase() -> None:
        for index in range(len(current_topic_words) - 1):
            phrase = " ".join(current_topic_words[index:index + 2])
            if phrase not in topics:
                topics.append(phrase)
        current_topic_words.clear()

    for word in words:
        if word in _STOP_WORDS:
            finish_topic_phrase()
            continue

        if word not in keywords:
            keywords.append(word)
        current_topic_words.append(word)

    finish_topic_phrase()
    return keywords, topics


def _classify_question(
    normalized_question: str, page_numbers: list[int]
) -> str:
    """Choose one question category using intent-specific rules in priority order."""
    # Explicit comparison intent wins over page-count or page-reference cues.
    if _COMPARISON_PATTERN.search(normalized_question):
        return "comparison"
    if _is_page_count_question(normalized_question):
        return "page_count"
    if page_numbers or re.search(r"\bpages?\b", normalized_question):
        return "page_lookup"
    if _SUMMARY_PATTERN.search(normalized_question):
        return "summary"
    if _EXPLANATION_PATTERN.search(normalized_question) or re.match(
        r"^(?:how (?:does|do|did|is|are|was|were|to)|help me understand)\b",
        normalized_question,
    ):
        return "explanation"
    if _FACTUAL_START_PATTERN.search(normalized_question):
        return "factual"
    return "general_question"


def analyze_question(
    question: str, total_pages: int | None = None
) -> dict[str, object]:
    """Classify a question and return page references, keywords, topics, and strategy.

    Page ranges are expanded into individual 1-based page numbers. If
    ``total_pages`` is supplied, references outside the document are omitted.
    Ranges longer than 1,000 pages are ignored to bound processing and memory.
    Empty or non-string input safely returns a general-question analysis.
    """
    if total_pages is not None and (
        isinstance(total_pages, bool)
        or not isinstance(total_pages, int)
        or total_pages <= 0
    ):
        raise ValueError("total_pages must be None or a positive integer.")

    if not isinstance(question, str):
        question = ""

    normalized_question = _normalize_question(question)
    page_numbers = _extract_page_numbers(question, total_pages)
    keywords, topics = _extract_keywords_and_topics(normalized_question)
    category = _classify_question(normalized_question, page_numbers)

    return {
        "category": category,
        "page_numbers": page_numbers,
        "keywords": keywords,
        "topics": topics,
        "strategy": _STRATEGIES[category],
    }
