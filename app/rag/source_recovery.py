"""Recover clause context from canonical PDF pages, never from guessed overlap."""
import re
from .source_display import evidence_blocks

_BOUNDARY = re.compile(
    r'(?m)^[ \t]*(?:(?P<item>\d{1,3}(?:\.\d{1,3}){1,4})(?:[.)]?[ \t]+(?![ \t]*[-–—])|(?=[A-Za-zÀ-ÿ]))'
    r'|\d{1,3}[.)]?[ \t]+(?=[A-ZÀ-Ý])|(?:ANEXO|CAP[ÍI]TULO|SE[ÇC][ÃA]O|CL[ÁA]USULA)\b)')


def _indexed(text):
    normalized, positions = [], []
    for match in re.finditer(r'\S+', text):
        if normalized:
            normalized.append(' ')
            positions.append(match.start() - 1)
        normalized.extend(match.group())
        positions.extend(range(match.start(), match.end()))
    return ''.join(normalized), positions


def _matches(text, needle):
    found, start = [], 0
    while needle and len(found) < 100:
        offset = text.find(needle, start)
        if offset < 0:
            break
        found.append(offset)
        start = offset + 1
    return found


def _readable(text):
    # Same deterministic reflow as the interface; no LLM and no paraphrase.
    from .topic_organizer import _source_line_records
    rows = [{'source_id': 'canonical', 'chunk_id': 0, 'page': 0, 'line': i + 1,
             'quote': record['text'], 'join_before': record['join_before']}
            for i, record in enumerate(_source_line_records(text))]
    return evidence_blocks(rows)[0]['text'] if rows else ''


class SourceRecovery:
    def __init__(self, pages, chunks, tables=None):
        self.pages = {i + 1: text for i, text in enumerate(pages)}
        self.index = {page: _indexed(text) for page, text in self.pages.items()}
        self.tables = {i + 1: rows for i, rows in enumerate(tables or [])}
        self.boundaries = {page: [match for match in _BOUNDARY.finditer(text)
            if not any(t["start"] <= match.start() < t["end"] for t in self.tables.get(page, []))]
            for page, text in self.pages.items()}
        self.chunks = {chunk['metadata']['chunk_id']: chunk for chunk in chunks}
        self.cache = {}
        self.body_ends = {}
        for page, text in self.pages.items():
            tail = list(re.finditer(r"(?m)^.*(?:p[áa]gina\s+\d+\s+(?:de|/)\s*\d+).*$", text, re.I))
            self.body_ends[page] = tail[-1].start() if tail and not text[tail[-1].end():].strip() else len(text)

    def _locate(self, evidence):
        page = evidence.get('page')
        normalized, positions = self.index.get(page, ('', []))
        quote = ' '.join(evidence.get('quote', '').split())
        matches = _matches(normalized, quote)
        if len(matches) > 1:
            # Repeated words alone cannot establish source identity. Narrow with
            # the full originating chunk, only if that chunk has one page match.
            chunk = self.chunks.get(evidence.get('chunk_id'), {})
            chunk_text = ' '.join(chunk.get('content', '').split())
            origins = _matches(normalized, chunk_text)
            if chunk_text and len(origins) == 1:
                left, right = origins[0], origins[0] + len(chunk_text)
                matches = [m for m in matches if left <= m and m + len(quote) <= right]
        if len(matches) != 1:
            return None
        start = matches[0]
        return positions[start], positions[start + len(quote) - 1] + 1

    def _ranges(self, page, start, end):
        text, boundaries = self.pages[page], self.boundaries[page]
        table = next((t for t in self.tables.get(page, []) if t["start"] <= start < t["end"]), None)
        if table:
            return [(table["start"], table["end"], None, False)]
        ranges = []
        for i, boundary in enumerate(boundaries):
            left = boundary.start()
            right = boundaries[i + 1].start() if i + 1 < len(boundaries) else self.body_ends[page]
            right = min([right] + [t["start"] for t in self.tables.get(page, []) if left < t["start"]])
            if boundary.group('item') and left < end and right > start:
                ranges.append((left, right, boundary.group('item'), True))
        if ranges and ranges[0][0] <= start:
            return ranges
        # Without an explicit numbered clause, use only the containing paragraph.
        separators = list(re.finditer(r'\n\s*\n', text))
        left = max([0] + [m.end() for m in separators if m.end() <= start])
        right = min([self.body_ends[page]] + [m.start() for m in separators if m.start() >= end])
        for boundary in boundaries:
            if boundary.start() <= start:
                left = max(left, boundary.start())
            elif boundary.start() >= end:
                right = min(right, boundary.start())
        if right - left > 6000:
            return [(start, end, None, False)]
        return [(left, right, None, False)]

    def _continuation_pair(self, page):
        if page - 1 not in self.pages or not self.boundaries.get(page - 1):
            return None
        previous = self.boundaries[page - 1][-1]
        if not previous.group("item"):
            return None
        previous_end = self.body_ends[page - 1]
        left = previous.start()
        head_end = self.boundaries[page][0].start() if self.boundaries.get(page) else self.body_ends[page]
        prefix = self.pages[page][:head_end].strip()
        tail = self.pages[page - 1][left:previous_end].strip()
        if (not prefix or not prefix[0].islower() or not tail or tail.endswith((".", ";", "!", "?"))
                or any(t["start"] < head_end for t in self.tables.get(page, []))):
            return None
        return (page - 1, left, previous_end, page, head_end, previous.group("item"))

    def _combined_clause(self, pair, source):
        previous, start, end, page, head_end, item = pair
        text = self.pages[previous][start:end].strip() + " " + self.pages[page][:head_end].strip()
        return {"page": previous, "pages": [previous, page], "start": start, "end": end,
                "segments": [{"page": previous, "start": start, "end": end},
                             {"page": page, "start": 0, "end": head_end}],
                "item": item, "text": _readable(text), "expanded": True, "source_ids": [source],
                "context_status": "clause_across_pages", "continuation_pending": False}

    def _linked_tables(self, page, table, source):
        while table.get("continued_from"):
            link = table["continued_from"]
            parent = next((t for t in self.tables.get(link["page"], []) if t["id"] == link["id"]), None)
            if parent is None or link["page"] >= page:
                break
            page, table = link["page"], parent
        pending, result, seen = [(page, table)], [], set()
        while pending:
            page, table = pending.pop(0)
            if (page, table.get("id")) in seen:
                continue
            seen.add((page, table.get("id")))
            result.append({"page": page, "start": table["start"], "end": table["end"],
                "item": None, "table": table, "text": _readable(self.pages[page][table["start"]:table["end"]]),
                "expanded": True, "source_ids": [source], "context_status": "table", "continuation_pending": False})
            for link in table.get("continuations", []):
                child = next((t for t in self.tables.get(link["page"], []) if t["id"] == link["id"]), None)
                if child is not None and link["page"] > page:
                    pending.append((link["page"], child))
        return result

    def _recover(self, evidence):
        key = (evidence.get('page'), evidence.get('chunk_id'), evidence.get('source_id'), evidence.get('quote'))
        if key in self.cache:
            return self.cache[key]
        location = self._locate(evidence)
        result = []
        if location is not None:
            page, source = evidence["page"], evidence.get("source_id")
            pair = self._continuation_pair(page)
            if pair and location[1] <= pair[4]:
                result = [self._combined_clause(pair, source)]
            else:
                for start, end, item, numbered in self._ranges(page, *location):
                    table = next((t for t in self.tables.get(page, []) if t["start"] == start and t["end"] == end), None)
                    if table:
                        result.extend(self._linked_tables(page, table, source))
                        continue
                    next_pair = self._continuation_pair(page + 1) if page + 1 in self.pages else None
                    if next_pair and next_pair[1] == start and numbered:
                        result.append(self._combined_clause(next_pair, source))
                        continue
                    if end - start > 12000:
                        start, end, item, numbered = *location, None, False
                    text = self.pages[page][start:end].strip()
                    result.append({'page': page, 'start': start, 'end': end, 'item': item,
                        'text': _readable(text), 'expanded': True, 'source_ids': [source],
                        'context_status': 'clause_on_page' if numbered else 'paragraph',
                        'continuation_pending': bool(numbered and end == self.body_ends[page]
                                                      and not text.endswith(('.', ';', '!', '?')))})
        self.cache[key] = result
        return result

    def enrich(self, topic_map):
        for theme in topic_map:
            for sub in theme.get('subtopics', []):
                blocks, fallback = {}, []
                for evidence in sub.get('evidence', []):
                    recovered = self._recover(evidence)
                    if not recovered:
                        fallback.append(evidence)
                    for row in recovered:
                        key = (row['page'], row['start'], row['end'])
                        if key not in blocks:
                            blocks[key] = {**row, 'source_ids': list(row['source_ids'])}
                        else:
                            blocks[key]['source_ids'].extend(source for source in row['source_ids']
                                if source not in blocks[key]['source_ids'])
                sub['source_blocks'] = sorted(blocks.values(), key=lambda row: (row['page'], row['start']))
                if "pages" in sub:
                    sub["pages"] = sorted(set(sub["pages"]) | {p for b in blocks.values() for p in b.get("pages", [b["page"]])})
                remaining, seen = [], set()
                for evidence in fallback:
                    page = evidence.get("page")
                    quote = " ".join(evidence.get("quote", "").split())
                    signature = (page, quote)
                    normalized, positions = self.index.get(page, ("", []))
                    matches = _matches(normalized, quote)
                    # Hide ambiguous repeated selections only when every occurrence
                    # is already represented by verified recovered page intervals.
                    covered = bool(matches) and all(any(any(segment["page"] == page and segment["start"] <= positions[m]
                        and positions[m + len(quote) - 1] < segment["end"]
                        for segment in b.get("segments", [b])) for b in blocks.values()) for m in matches)
                    if not covered and signature not in seen:
                        remaining.append(evidence)
                        seen.add(signature)
                for block in evidence_blocks(remaining):
                    sub['source_blocks'].append({**block, 'context_status': 'selected_only', 'expanded': False})
        return topic_map
