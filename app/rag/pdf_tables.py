"""Geometry-backed table extraction; retain merged cells without inferring values."""
import html
import re


def is_tabular(rows):
    """Reject single-column prose and grids consisting of clause labels + prose."""
    width = max((len(row) for row in rows), default=0)
    columns = [[str(row[i] or "").strip() for row in rows if i < len(row) and row[i]] for i in range(width)]
    active = [col for col in columns if len(col) >= 2]
    if len(active) < 2:
        return False
    labels = re.compile(r"^\d+(?:\.\d+)+[.)]?$")
    prose = [col for col in active if not all(labels.fullmatch(cell) for cell in col)]
    if len(prose) == 1 and sum(len(cell) for cell in prose[0]) / len(prose[0]) > 85:
        return False
    return True


def link_table_continuations(pages):
    """Connect adjacent page-edge grids only with aligned columns and no new title."""
    for index in range(len(pages) - 1):
        if not pages[index] or not pages[index + 1]:
            continue
        parent, child = pages[index][-1], pages[index + 1][0]
        a, b = parent.get("bbox"), child.get("bbox")
        if not a or not b or child["title"] != "Tabela do PDF":
            continue
        if (parent.get("page_height", 0) - a[3] <= 100 and b[1] <= 80
                and parent["column_count"] == child["column_count"]
                and abs(a[0] - b[0]) < 3 and abs(a[2] - b[2]) < 3):
            parent.setdefault("continuations", []).append({"page": index + 2, "id": child["id"]})
            child["continued_from"] = {"page": index + 1, "id": parent["id"]}
            child["title"] = parent["title"].split(" · continuação")[0] + " · continuação"
    return pages


def extract_page_tables(page, text, clean):
    lines, cursor = [], 0
    for line in page.extract_text_lines():
        value = clean(line['text'], preserve_lines=True)
        offset = text.find(value, cursor) if value else -1
        if offset >= 0:
            lines.append({**line, 'start': offset, 'end': offset + len(value)})
            cursor = offset + len(value)
    output = []
    for index, table in enumerate(page.find_tables()):
        rows = table.extract()
        if not is_tabular(rows):
            continue
        x0, top, x1, bottom = table.bbox
        inside = [line for line in lines if top <= (line['top'] + line['bottom']) / 2 <= bottom]
        if not inside:
            continue
        start = inside[0]['start']
        following = [line['start'] for line in lines if line['top'] >= bottom - 0.1 and line['start'] >= start]
        end = min(following, default=len(text))
        preceding = [line for line in lines if line['bottom'] <= top and top - line['bottom'] < 50]
        title = None
        for line in reversed(preceding):
            match = re.search(r'\b(?:TABELA|QUADRO)\s+[\w.\-]+', line['text'], re.I)
            if match:
                title = match.group()
                break
        xs = sorted({cell[0] for cell in table.cells})
        tops = [min(cell[1] for cell in row.cells if cell is not None) for row in table.rows]
        cells = []
        for row_index, row in enumerate(table.rows):
            for col_index, cell in enumerate(row.cells):
                if cell is None:
                    continue
                cells.append({'row': row_index, 'col': col_index,
                    'rowspan': max(1, sum(cell[1] - 0.1 <= y < cell[3] - 0.1 for y in tops)),
                    'colspan': max(1, sum(cell[0] - 0.1 <= x < cell[2] - 0.1 for x in xs)),
                    'text': (rows[row_index][col_index] or '').strip()})
        output.append({'id': f'table-{index}', 'start': start, 'end': end,
                       'title': title or 'Tabela do PDF', 'rows': rows, 'cells': cells,
                       'row_count': len(rows), 'column_count': len(xs),
                       'bbox': list(table.bbox), 'page_height': getattr(page, 'height', 0)})
    return output


def table_html(table):
    by_row = {}
    for cell in table['cells']:
        by_row.setdefault(cell['row'], []).append(cell)
    rows = []
    for index in range(table['row_count']):
        cells = []
        for cell in sorted(by_row.get(index, []), key=lambda c: c['col']):
            cells.append(f'<td rowspan="{int(cell["rowspan"])}" colspan="{int(cell["colspan"])}" '
                         'style="border:1px solid #8b949e;padding:8px;vertical-align:top;white-space:pre-wrap;min-width:85px">'
                         + html.escape(cell['text']) + '</td>')
        rows.append('<tr>' + ''.join(cells) + '</tr>')
    return ('<div style="overflow:auto;max-height:650px;margin:8px 0"><table style="border-collapse:collapse;width:100%;font-size:0.9rem">'
            + ''.join(rows) + '</table></div>')
