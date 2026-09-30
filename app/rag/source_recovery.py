"""Recover clause context from canonical PDF pages, never from guessed overlap."""
import re
from .source_display import evidence_blocks

_BOUNDARY = re.compile(
    r'(?m)^[ \t]*(?:(?P<item>\d{1,3}(?:\.\d{1,3}){1,4})(?:[.)]?[ \t]+|(?=[A-Za-zÀ-ÿ]))'
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
    def __init__(self, pages, chunks):
        self.pages = {i + 1: text for i, text in enumerate(pages)}
        self.index = {page: _indexed(text) for page, text in self.pages.items()}
        self.boundaries = {page: list(_BOUNDARY.finditer(text)) for page, text in self.pages.items()}
        self.chunks = {chunk['metadata']['chunk_id']: chunk for chunk in chunks}
        self.cache = {}

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
        ranges = []
        for i, boundary in enumerate(boundaries):
            left = boundary.start()
            right = boundaries[i + 1].start() if i + 1 < len(boundaries) else len(text)
            if boundary.group('item') and left < end and right > start:
                ranges.append((left, right, boundary.group('item'), True))
        if ranges and ranges[0][0] <= start:
            return ranges
        # Without an explicit numbered clause, use only the containing paragraph.
        separators = list(re.finditer(r'\n\s*\n', text))
        left = max([0] + [m.end() for m in separators if m.end() <= start])
        right = min([len(text)] + [m.start() for m in separators if m.start() >= end])
        for boundary in boundaries:
            if boundary.start() <= start:
                left = max(left, boundary.start())
            elif boundary.start() >= end:
                right = min(right, boundary.start())
        if right - left > 6000:
            return [(start, end, None, False)]
        return [(left, right, None, False)]

    def _recover(self, evidence):
        key = (evidence.get('page'), evidence.get('chunk_id'), evidence.get('source_id'), evidence.get('quote'))
        if key in self.cache:
            return self.cache[key]
        location = self._locate(evidence)
        if location is None:
            result = []
        else:
            result = []
            for start, end, item, numbered in self._ranges(evidence['page'], *location):
                # Bound reconstruction rather than claiming an enormous section
                # is a complete clause when its boundary cannot be established.
                if end - start > 12000:
                    start, end, item, numbered = *location, None, False
                text = self.pages[evidence['page']][start:end].strip()
                result.append({'page': evidence['page'], 'start': start, 'end': end,
                    'item': item, 'text': _readable(text), 'expanded': True,
                    'source_ids': [evidence.get('source_id')],
                    'context_status': 'clause_on_page' if numbered else 'paragraph',
                    'continuation_pending': bool(numbered and end == len(self.pages[evidence['page']])
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
                for block in evidence_blocks(fallback):
                    sub['source_blocks'].append({**block, 'context_status': 'selected_only', 'expanded': False})
        return topic_map
