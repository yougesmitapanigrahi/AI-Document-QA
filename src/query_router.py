"""Select document chunks for a question analysis result."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any


def retrieve_chunks_for_analysis(
    analysis: Mapping[str, Any],
    chunks: Sequence[Mapping[str, Any]],
    semantic_search: Callable[[], Sequence[Mapping[str, Any]]],
) -> list[Mapping[str, Any]]:
    """Choose page-filtered chunks or call the supplied semantic search.

    Page-count questions need no chunks. Explicit page lookups use only chunks
    from requested pages. Other categories currently use semantic search.
    """
    category = analysis["category"]
    if category == "page_count":
        return []

    if category == "page_lookup" and analysis["page_numbers"]:
        requested_pages = set(analysis["page_numbers"])
        return [
            chunk
            for chunk in chunks
            if chunk.get("page_number") in requested_pages
        ]

    return list(semantic_search())
