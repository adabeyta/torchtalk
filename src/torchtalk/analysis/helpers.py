"""Shared helper functions for TorchTalk analyzers."""

from typing import Any, Dict, List, Set, Tuple


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def fuzzy_match(query: str, candidates: List[str], max_results: int = 10) -> List[str]:
    """Find candidates matching query with fuzzy matching."""
    query_lower = query.lower()
    results: List[Tuple[int, int, str]] = []  # (priority, length, name)

    # Exact matches (priority 0)
    exact = {c for c in candidates if query_lower == c.lower()}
    results.extend((0, len(c), c) for c in exact)

    # Substring matches (priority 1)
    substring = {c for c in candidates if query_lower in c.lower() and c not in exact}
    results.extend((1, len(c), c) for c in substring)

    # Levenshtein matches (priority = 2 + distance)
    if len(query) >= 3:
        checked = exact | substring
        max_dist = max(3, len(query) // 2)
        for candidate in candidates:
            if candidate in checked or abs(len(candidate) - len(query)) > 5:
                continue
            dist = levenshtein_distance(query_lower, candidate.lower())
            if dist <= max_dist:
                results.append((2 + dist, len(candidate), candidate))

    results.sort(key=lambda x: (x[0], x[1]))
    seen: Set[str] = set()
    return [n for _, _, n in results if not (n in seen or seen.add(n))][:max_results]


def safe_sort_key(item: Any) -> str:
    """Sort key that handles None values (sorts None last)."""
    if item is None:
        return "\uffff"
    return str(item) if not isinstance(item, str) else item


def truncate(text: str, max_len: int = 80) -> str:
    """Truncate text with ellipsis."""
    if not text or len(text) <= max_len:
        return text or ""
    return text[: max_len - 3] + "..."


def dedupe_by_key(items: List[Dict], key: str) -> List[Dict]:
    """Deduplicate list of dicts by a key."""
    seen: Set[Any] = set()
    result = []
    for item in items:
        if (val := item.get(key)) and val not in seen:
            seen.add(val)
            result.append(item)
    return result
