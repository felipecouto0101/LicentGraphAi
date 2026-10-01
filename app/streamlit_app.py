"""Etapa atual: organização e navegação do mapa, sem chat ou explicações."""
import time
import re
import html
import logging
from concurrent.futures import ThreadPoolExecutor
from app.ui_health import APIHealth
import requests
import streamlit as st

logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="LicitGraphAi · Assuntos do edital", page_icon="🧭", layout="wide")
API_URL = "http://127.0.0.1:8002"
logger = logging.getLogger(__name__)


@st.cache_resource
def _health_executor():
    return ThreadPoolExecutor(max_workers=4, thread_name_prefix="ui-health")


@st.fragment(run_every=1)
def _connection_status(api_url):
    health = st.session_state.get("api_health")
    if health is None:
        health = st.session_state["api_health"] = APIHealth()
    snapshot = health.poll(api_url, _health_executor())
    if snapshot["state"] == "checking":
        st.caption("Verificando conexão com a API…")
    elif snapshot["state"] == "unavailable":
        st.warning("API indisponível. Inicie o backend para enviar um PDF.")
    else:
        status = snapshot["status"]
        if not status.get("ready", True) or status.get("analysis_mode") == "demo":
            st.warning("Configure a chave do provedor e desative o modo demonstração.")
        else:
            st.caption("API disponível")


def _api_error(response):
    try:
        return response.json().get("detail", "Não foi possível concluir a solicitação.")
    except ValueError:
        return f"A API respondeu com erro {response.status_code}."


def _reset():
    for key in ("result", "job_id", "topic_search", "theme_choice", "selected_theme", "conversation", "selected_topic"):
        st.session_state.pop(key, None)


def _prepare(api_url, file):
    try:
        response = requests.post(f"{api_url}/analyze/upload",
            files={"file": (file.name, file.getvalue(), "application/pdf")}, timeout=40)
        if response.status_code != 200:
            st.error(_api_error(response))
            return
        _reset()
        st.session_state["job_id"] = response.json()["job_id"]
        st.rerun()
    except requests.RequestException:
        st.error("Não foi possível enviar o PDF. Confira se a API está em execução.")


@st.fragment(run_every=3)
def _poll(api_url, job_id):
    try:
        response = requests.get(f"{api_url}/job/{job_id}", timeout=8)
        if response.status_code != 200:
            st.error(_api_error(response))
            if st.button("Enviar novamente"):
                _reset()
                st.rerun()
            return
        job = response.json()
    except requests.RequestException:
        st.warning("A conexão com a API foi interrompida. O processamento pode continuar no backend.")
        if st.button("Verificar novamente"):
            st.rerun()
        return
    if job["status"] == "done":
        result = job.get("result") or {}
        if result.get("map_version") != 4 or not result.get("topic_map"):
            st.error("Atualize e reinicie a aplicação para preparar o novo mapa.")
            if st.button("Voltar ao envio"):
                _reset()
                st.rerun()
            return
        result["filename"] = job.get("file", "Edital em PDF")
        st.session_state["result"] = result
        st.rerun()
    elif job["status"] == "error":
        st.error(job.get("error") or "Não foi possível ler o PDF.")
        if st.button("Voltar ao envio"):
            _reset()
            st.rerun()
        return
    progress = job.get("progress", {})
    with st.container(border=True):
        st.subheader("Preparando seu mapa")
        st.write(progress.get("stage", "Aguardando processamento"))
        total = progress.get("total", 0)
        done = progress.get("completed", 0)
        if total:
            st.progress(min(done / total, 1), text=f"{done} de {total} etapas de organização concluídas")
        activity = progress.get("activity_message")
        if activity:
            if progress.get("activity") in {"waiting", "retrying", "repairing", "splitting"}:
                st.warning(activity)
            else:
                st.info(activity)
        wait_until = progress.get("wait_until")
        if isinstance(wait_until, (int, float)):
            seconds = max(0, int(wait_until - time.time() + 0.999))
            st.caption(f"Próxima tentativa em aproximadamente {seconds} segundos.")
        if progress.get("provider"):
            st.caption(f"Provedor: {progress['provider']} · Modelo: {progress.get('model', '')}")
        st.caption("A IA organiza apenas nomes e agrupamentos. As cotas da API podem exigir pausas entre chamadas.")


def _retry(api_url, job_id):
    try:
        response = requests.post(f"{api_url}/job/{job_id}/organize-map", timeout=10)
        if response.status_code not in (200, 409):
            st.error(_api_error(response))
            return
        st.session_state.pop("result", None)
        st.rerun()
    except requests.RequestException:
        st.warning("Não foi possível confirmar a retomada. Verifique o andamento antes de reenviar o PDF.")


def _pages(pages):
    values = sorted(set(pages))
    ranges = []
    for value in values:
        if ranges and value == ranges[-1][1] + 1:
            ranges[-1][1] = value
        else:
            ranges.append([value, value])
    return ", ".join(str(a) if a == b else f"{a}–{b}" for a, b in ranges)


def _visible_themes(themes, search):
    search = search.casefold().strip()
    visible = []
    for theme in themes:
        children = [sub for sub in theme["subtopics"] if not search or search in
                    (theme["title"] + " " + sub["title"] + " " + " ".join(sub.get("source_titles", []))).casefold()]
        if children:
            visible.append((theme, children))
    return visible


def _count_label(count, organized):
    singular, plural = ("assunto", "assuntos") if organized else ("subseção", "subseções")
    return f"{count} {singular if count == 1 else plural}"


def _subtopic_card(sub, organized, source_key="sources"):
    # Source imports and rendering run only after opening this subject.
    generic = sub["title"] in ("Visão geral da seção", "Visão geral e cláusulas da seção")
    with st.container(border=True):
        st.subheader("Conteúdo da seção" if generic and not organized else sub["title"])
        st.caption("Páginas do PDF: " + _pages(sub["pages"]))
        if generic and not organized:
            st.caption("Esta seção ainda não foi dividida em assuntos pela IA.")
        evidence = sub.get("evidence", [])
        if evidence:
            sources = st.expander("Conferir trechos de origem", key=source_key, on_change="rerun")
            if not sources.open:
                return
            from app.rag.source_display import evidence_blocks
            from app.rag.pdf_tables import table_html
            with sources:
                blocks = sub.get("source_blocks")
                if blocks is None:
                    blocks = evidence_blocks(evidence)
                for block in blocks:
                    with st.container(border=True):
                        label = ("Fonte: páginas " + _pages(block["pages"]) if len(block.get("pages", [])) > 1
                                 else f"Fonte: página {block['page']}" if block["page"] is not None else "Fonte: trecho do PDF")
                        if block.get("item"):
                            label += f" · item {block['item']}"
                        st.caption(label)
                        if block.get("table"):
                            st.subheader(block["table"]["title"])
                            st.caption("Tabela extraída do PDF; células mescladas preservadas quando identificadas.")
                            st.markdown(table_html(block["table"]), unsafe_allow_html=True)
                            continue
                        if block.get("expanded"):
                            st.caption("Item recuperado entre páginas consecutivas do PDF." if block.get("context_status") == "clause_across_pages"
                                       else "Contexto recuperado do texto original do PDF.")
                        elif block.get("context_status") == "selected_only":
                            st.caption("Trecho selecionado; não foi possível recuperar a continuação com segurança.")
                        if block.get("continuation_pending"):
                            st.caption("O item pode continuar na página seguinte; este bloco contém somente a página indicada.")
                        # Escape document content; preserve lists and paragraph spacing
                        # using the interface font instead of rendering PDF text as Markdown.
                        st.markdown('<div style="white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.65">'
                                    + html.escape(block["text"]) + '</div>', unsafe_allow_html=True)


def _theme_label(title):
    return re.sub(r"^(?:das|dos|da|do)\s+", "", title.strip(), flags=re.I)


def _topic_browser(api_url, job_id, result):
    themes = result["topic_map"]
    complete = result.get("organization_status") == "done"
    partial = result.get("organization_status") == "partial"
    organized = complete or partial
    with st.container(border=True):
        st.caption(result.get("filename", "Edital em PDF"))
        st.header("Mapa parcial de assuntos" if partial else "Assuntos do seu edital" if complete else "Índice original do PDF")
        st.caption("Subtemas validados · organização incompleta" if partial else "Organizado com IA" if complete
                   else "Organização com IA interrompida · índice estrutural disponível")
        columns = st.columns(3)
        columns[0].metric("Temas" if organized else "Seções", len(themes))
        columns[1].metric("Assuntos" if organized else "Subseções identificadas",
            sum(1 for t in themes for sub in t["subtopics"] if organized or sub["title"] not in
                ("Visão geral da seção", "Visão geral e cláusulas da seção", "Dados de abertura do edital")))
        columns[2].metric("Páginas", result.get("page_count") or "—")
    if not complete:
        with st.container(border=True):
            st.subheader("Organização pendente")
            st.warning("A organização foi interrompida. Os subtemas já validados estão disponíveis abaixo; o restante ainda está pendente."
                       if partial else "A IA ainda não produziu subtemas validados. Abaixo aparece somente o índice original do PDF.")
            with st.expander("Detalhes da interrupção", expanded=True):
                st.write(result.get("organization_error") or "Organização pendente.")
            if st.button("Retomar organização com IA", type="primary"):
                _retry(api_url, job_id)
    with st.container(border=True):
        st.subheader("Mapa de assuntos" if organized else "Navegação pelas seções do PDF")
        st.write("Os subtemas aparecem dentro de cada tema. Abra ou feche os blocos para explorar o mapa." if organized
                 else "Estas são seções do documento; não representam subtemas gerados pela IA.")
        search = st.text_input("Buscar um assunto" if organized else "Buscar uma seção",
                              placeholder="Ex.: documentos, proposta, pagamento", key="topic_search")
        visible = _visible_themes(themes, search)
        if not visible:
            st.info("Nenhum resultado encontrado. Tente outra palavra.")
        else:
            st.caption(f"{len(visible)} de {len(themes)} " + ("temas" if organized else "seções"))
            for index, (theme, children) in enumerate(visible):
                label = _theme_label(theme["title"])
                real_children = [sub for sub in children if organized or sub["title"] not in
                                 ("Visão geral da seção", "Visão geral e cláusulas da seção", "Dados de abertura do edital")]
                count = _count_label(len(real_children), organized) if real_children else "subtemas pendentes"
                theme_box = st.expander(f"{label} · {count}", expanded=bool(search) or index == 0,
                                        key=f"theme:{job_id}:{theme['title']}", on_change="rerun")
                if not theme_box.open:
                    continue
                with theme_box:
                    if label != theme["title"]:
                        st.caption("Título de origem: " + theme["title"])
                    if not real_children:
                        pages = sorted({page for sub in children for page in sub["pages"]})
                        st.caption("Páginas do PDF: " + _pages(pages))
                        st.info("Subtemas ainda não identificados para esta seção.")
                    else:
                        for child_index, sub in enumerate(real_children):
                            _subtopic_card(sub, organized, source_key=f"sources:{job_id}:{theme['title']}:{child_index}")
        st.caption("As páginas indicam a origem das informações. O mapa ainda não gera explicações.")
    missing = [r for r in result.get("annex_references", []) if r["status"] == "not_located"]
    if missing:
        with st.container(border=True):
            st.subheader("Conferência de anexos")
            with st.expander(f"Referências sem seção identificada · {len(missing)}"):
                st.write("Estes anexos foram mencionados, mas não identificados como seções neste arquivo. Podem estar em arquivos separados ou ter outro formato de título.")
                for reference in missing:
                    st.caption(reference["label"] + " · mencionado nas páginas " + _pages(reference["pages"]))


def main():
    started = time.perf_counter()
    st.title("🧭 LicitGraphAi")
    st.write("Um mapa organizado para entender o que seu edital aborda.")
    stored = st.session_state.get("result")
    if stored and stored.get("map_version") != 4:
        _reset()
        st.info("Envie novamente o PDF para organizar os assuntos com IA.")
    with st.sidebar:
        st.header("Seu documento")
        with st.expander("Conexão com a API"):
            api_url = st.text_input("Endereço", value=API_URL).rstrip("/")
        file = st.file_uploader("Enviar edital", type="pdf")
        busy = st.session_state.get("job_id") and not st.session_state.get("result")
        if st.button("Organizar assuntos", type="primary", disabled=file is None or bool(busy), use_container_width=True):
            _prepare(api_url, file)
        st.caption("Nesta etapa, a IA cria temas e subtemas. Explicações e chat serão adicionados depois.")
    if st.session_state.get("result") and st.session_state.get("job_id"):
        _topic_browser(api_url, st.session_state["job_id"], st.session_state["result"])
    elif st.session_state.get("job_id"):
        _poll(api_url, st.session_state["job_id"])
    else:
        with st.container(border=True):
            st.subheader("Comece pelo seu edital")
            st.write("Envie um PDF na lateral. Vamos identificar suas seções e organizar os assuntos em nomes claros.")
            st.caption("Cada assunto mantém as páginas e seções de origem para você consultar o documento.")

    logger.info("ui_render duration_ms=%.1f has_result=%s", (time.perf_counter() - started) * 1000,
                bool(st.session_state.get("result")))
    # Render the shell first; the network check runs in a worker and only this
    # fragment updates as the result becomes available.
    with st.sidebar:
        _connection_status(api_url)


if __name__ == "__main__":
    main()
