"""Join verified consecutive evidence without inventing missing source lines."""
import re

_START = re.compile(r'^(?:\d+(?:\.\d+)+(?:\s|(?=[A-Za-zÀ-ÿ]))|\d+(?:\.\d+)*\.?\s|[a-zA-Z]\)|[IVXLCDM]+[.)]\s|[•●▪-]\s)')


def _join(previous, current):
    separator = current.get('join_before')
    if separator is None:  # Maps generated before source layout metadata.
        return '\n\n' if _START.match(current['quote'].lstrip()) else ' '
    if '\n' in separator or '\r' in separator:
        if separator.count('\n') > 1 or _START.match(current['quote'].lstrip()):
            return '\n\n'
        return ' '
    return separator


def evidence_blocks(evidence):
    """Separate pages/sources and gaps; order selected lines within each source."""
    sources = {}
    for item in evidence:
        key = (item.get('page'), item.get('source_id'), item.get('chunk_id'))
        sources.setdefault(key, []).append(item)
    result = []
    for (page, source_id, chunk_id), items in sources.items():
        numbered = all(type(item.get('line')) is int for item in items)
        ordered = sorted(items, key=lambda item: item['line']) if numbered else items
        seen, block, previous = set(), None, None
        for item in ordered:
            signature = (item.get('line'), item.get('quote', ''))
            if signature in seen:
                continue
            seen.add(signature)
            contiguous = (block is not None and numbered and item['line'] == previous['line'] + 1)
            if not contiguous:
                block = {'page': page, 'source_id': source_id, 'chunk_id': chunk_id,
                         'first_line': item.get('line'), 'last_line': item.get('line'), 'text': item['quote']}
                result.append(block)
            else:
                block['text'] += _join(previous, item) + item['quote']
                block['last_line'] = item['line']
            previous = item
    return result
