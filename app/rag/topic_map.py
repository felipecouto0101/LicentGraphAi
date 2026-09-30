"""Índice navegável do PDF, construído sem gerar explicações."""

from __future__ import annotations

import re
import unicodedata
from collections import OrderedDict


_NUMBERED = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2}){0,3})\s*[.\-)–:]?\s+(.{4,110})$")
_CLAUSE = re.compile(r"^\s*(?:CAP[ÍI]TULO|SE[ÇC][ÃA]O|ANEXO|PARTE)\s+([IVXLCDM\d]+)\s*[-–:.]?\s*(.{4,100})$", re.I)
_NOISE = re.compile(r"^(?:p[áa]gina\s*\d+|\d+\s*/\s*\d+|edital\s+n[º°o]?)$", re.I)


def _clean(value: str) -> str:
    return " ".join(re.sub(r"^[\s\-–:.]+|[\s\-–:.]+$", "", value).split())


def _key(value: str) -> str:
    plain = unicodedata.normalize("NFKD", value.casefold())
    return re.sub(r"\W+", " ", "".join(c for c in plain if not unicodedata.combining(c))).strip()


def _display(value: str) -> str:
    return value.title() if value.isupper() else value


def _heading(content: str):
    """Identifica apenas cabeçalhos observáveis; jamais inventa um assunto."""
    for line in content.splitlines()[:6]:
        line = _clean(line)
        if not line or _NOISE.match(line) or len(line) > 115:
            continue
        clause = _CLAUSE.match(line)
        if clause:
            return clause.group(1), _clean(clause.group(2)), 1
        numbered = _NUMBERED.match(line)
        if numbered:
            title = _clean(numbered.group(2))
            if len(title.split()) <= 13 and not title.endswith((";", ",")):
                return numbered.group(1), title, numbered.group(1).count(".") + 1
        if 7 <= len(line) <= 85 and len(line.split()) <= 11 and line.isupper():
            return None, line.title(), 1
    return None


def build_topic_map(chunks: list[dict]) -> list[dict]:
    """Agrupa todos os chunks; repetição de título reúne as fontes sem perda."""
    themes: OrderedDict[str, dict] = OrderedDict()
    current_theme = None
    current_subtopic = None
    for index, chunk in enumerate(chunks):
        content = chunk.get("content", "").strip()
        if not content:
            continue
        heading = _heading(content)
        page = chunk.get("metadata", {}).get("page")
        chunk_id = chunk.get("metadata", {}).get("chunk_id", index)
        if heading:
            numbering, title, level = heading
            if level == 1:
                major_key = _key(title)
                if major_key not in themes:
                    themes[major_key] = {"id": f"tema-{len(themes) + 1}",
                                         "title": title, "display_title": _display(title),
                                         "subtopics": []}
                current_theme = themes[major_key]
                current_subtopic = None
            else:
                # O primeiro nível numerado identifica o grupo em documentos
                # cujas subseções aparecem antes de um cabeçalho principal.
                if current_theme is None:
                    major_key = f"secao-{numbering.split('.')[0]}"
                    themes.setdefault(major_key, {"id": f"tema-{len(themes) + 1}",
                                                   "title": f"Seção {numbering.split('.')[0]}",
                                                   "subtopics": []})
                    current_theme = themes[major_key]
                current_subtopic = None
            if level == 1 and len(content.splitlines()) > 1:
                # O texto que segue um título também pertence ao assunto.
                current_subtopic = title
            elif level > 1:
                current_subtopic = title
        if current_theme is None:
            themes.setdefault("sem-cabecalho", {"id": "tema-inicial",
                                                 "title": "Introdução e dados iniciais",
                                                 "subtopics": []})
            current_theme = themes["sem-cabecalho"]
        sub_title = current_subtopic or "Visão geral e cláusulas da seção"
        existing = next((item for item in current_theme["subtopics"]
                         if _key(item["title"]) == _key(sub_title)), None)
        if existing is None:
            existing = {"id": f"subtema-{sum(len(t['subtopics']) for t in themes.values()) + 1}",
                        "title": sub_title, "display_title": _display(sub_title),
                        "pages": [], "chunk_ids": []}
            current_theme["subtopics"].append(existing)
        if page is not None and page not in existing["pages"]:
            existing["pages"].append(page)
        if chunk_id not in existing["chunk_ids"]:
            existing["chunk_ids"].append(chunk_id)
    return list(themes.values())


def all_chunk_ids(topic_map: list[dict]) -> set:
    return {chunk_id for theme in topic_map for sub in theme["subtopics"]
            for chunk_id in sub["chunk_ids"]}


def merge_similar_themes(topic_map: list[dict], encoder, threshold: float = 0.84) -> list[dict]:
    """Reúne títulos semanticamente próximos, mantendo os subtemas e as fontes."""
    import numpy as np

    if len(topic_map) < 2:
        return topic_map
    vectors = np.asarray(encoder.encode([theme["title"] for theme in topic_map]), dtype=float)
    norms = np.linalg.norm(vectors, axis=1)
    grouped: list[dict] = []
    representatives = []
    for index, theme in enumerate(topic_map):
        generic = theme["title"].startswith("Seção ") or theme["id"] == "tema-inicial"
        best = None
        if not generic and norms[index]:
            for position, other in enumerate(representatives):
                if norms[other] and float(vectors[index] @ vectors[other] /
                                          (norms[index] * norms[other])) >= threshold:
                    best = position
                    break
        if best is None:
            grouped.append(theme)
            representatives.append(index)
            continue
        target = grouped[best]
        target.setdefault("related_titles", []).append(theme["title"])
        for sub in theme["subtopics"]:
            match = next((candidate for candidate in target["subtopics"]
                          if _key(candidate["title"]) == _key(sub["title"])), None)
            if match is None:
                target["subtopics"].append(sub)
            else:
                for field in ("pages", "chunk_ids"):
                    match[field].extend(value for value in sub[field] if value not in match[field])
    return grouped
