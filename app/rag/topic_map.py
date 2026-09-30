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


def split_topic_sections(pages: list[str]) -> list[dict]:
    """Lê cabeçalhos em todas as linhas ANTES de fragmentar ou reordenar texto."""
    blocks = []
    theme, subtopic = "Introdução e dados iniciais", "Dados de abertura do edital"
    # Não tratar frases em caixa alta da capa nem regras numeradas como títulos.
    numbered = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2}){0,3})[.)]?\s+(.+)$")
    annex = re.compile(r"^ANEXO\s+([IVXLCDM]+|\d+)(?:\s*[-–:]\s*(.*))?$", re.I)
    clause = re.compile(r"^CL[ÁA]USULA\s+(.+?)\s*[-–:]\s*(.+)$", re.I)
    body_verbs = re.compile(r"\b(?:devera|deverao|deve|podera|poderao|sera|serao|fica|sao)\b")
    in_annex = False
    for page_number, page in enumerate(pages, 1):
        lines = page.splitlines()
        pending = []

        def flush():
            content = "\n".join(pending).strip()
            if content:
                blocks.append({"content": content, "page": page_number,
                               "topic_title": theme, "subtopic_title": subtopic})
            pending.clear()

        is_contents = any(_key(line) in {"sumario", "indice", "indice geral"} for line in lines[:6])
        index = 0
        while index < len(lines):
            line = lines[index]
            clean = _clean(line)
            match = numbered.match(clean)
            ann = annex.match(clean)
            cl = clause.match(clean)
            kind, title = None, None
            # Pontilhados são entradas do sumário, não seções do corpo.
            if not is_contents and "..." not in clean:
                if ann:
                    kind, title = "theme", clean
                    in_annex = True
                elif cl:
                    kind, title = ("sub" if in_annex else "theme"), cl.group(2)
                elif (match and ("." in match.group(1)
                                 or re.match(r"^\d+(?:\.\d+)*[.)]\s", clean)
                                 or (match.group(1).isdigit() and not in_annex))
                      and match.group(2).isupper() and not body_verbs.search(_key(match.group(2)))):
                    kind = "theme" if "." not in match.group(1) and not in_annex else "sub"
                    title = match.group(2)
                elif _key(clean) == "preambulo":
                    kind, title = "theme", clean
            consumed = 1
            if kind:
                # Continuações em caixa alta pertencem ao mesmo título.
                while index + consumed < len(lines):
                    following = _clean(lines[index + consumed])
                    if (not following or not following.isupper() or "..." in following
                            or numbered.match(following) or annex.match(following)
                            or clause.match(following) or following.endswith(":")
                            or len(title + " " + following) > 180):
                        break
                    # Só anexos sem descrição aceitam o título na linha seguinte;
                    # cabeçalhos completos não absorvem parágrafos em caixa alta.
                    if not (consumed == 1 and ann and not ann.group(2)) and not title.endswith((" DE", " E", " DA", " DO", " DAS", " DOS")):
                        break
                    title += " " + following
                    consumed += 1
                flush()
                if kind == "theme":
                    theme, subtopic = _display(_clean(title)), "Visão geral da seção"
                else:
                    subtopic = _display(_clean(title))
            pending.extend(lines[index:index + consumed])
            index += consumed
        flush()
    return blocks


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
        structural = chunk.get("metadata", {})
        if structural.get("topic_title"):
            title = structural["topic_title"]
            key = _key(title)
            if key not in themes:
                themes[key] = {"id": f"tema-{len(themes) + 1}", "title": title,
                               "display_title": _display(title), "subtopics": []}
            current_theme = themes[key]
            current_subtopic = structural.get("subtopic_title") or "Visão geral da seção"
            heading = None
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
        generic = (theme["title"].startswith("Seção ")
                   or theme["title"] == "Introdução e dados iniciais")
        best = None
        if not generic and norms[index]:
            for position, other in enumerate(representatives):
                if (topic_map[other]["title"].startswith("Seção ")
                        or topic_map[other]["title"] == "Introdução e dados iniciais"):
                    continue
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
