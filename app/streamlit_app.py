"""Etapa atual: organização e navegação do mapa, sem chat ou explicações."""
import time
import requests
import streamlit as st

st.set_page_config(page_title="LicitGraphAi · Assuntos do edital", page_icon="🧭", layout="wide")
API_URL = "http://127.0.0.1:8002"


def _api_error(response):
    try:
        return response.json().get("detail", "Não foi possível concluir a solicitação.")
    except ValueError:
        return f"A API respondeu com erro {response.status_code}."


def _reset():
    for key in ("result", "job_id", "topic_search", "selected_theme", "conversation", "selected_topic"):
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
        if result.get("map_version") != 3 or not result.get("topic_map"):
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
        st.caption("A IA organiza apenas nomes e agrupamentos. As cotas da API podem exigir pausas entre chamadas.")
    time.sleep(3)
    st.rerun()


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


def _topic_browser(api_url, job_id, result):
    themes = result["topic_map"]
    organized = result.get("organization_status") == "done"
    st.caption(result.get("filename", "Edital em PDF"))
    st.header("Assuntos do seu edital" if organized else "Índice original do PDF")
    st.write("Explore os temas e veja onde cada assunto aparece no documento.")
    columns = st.columns(3)
    columns[0].metric("Temas", len(themes))
    columns[1].metric("Assuntos", sum(len(t["subtopics"]) for t in themes))
    columns[2].metric("Páginas", result.get("page_count") or "—")
    if not organized:
        st.warning("A IA ainda não concluiu a organização. Abaixo está o índice original, preservado para conferência.")
        with st.expander("Detalhes da interrupção"):
            st.write(result.get("organization_error") or "Organização pendente.")
        if st.button("Retomar organização com IA", type="primary"):
            _retry(api_url, job_id)
    search = st.text_input("Buscar um assunto", placeholder="Ex.: documentos, proposta, pagamento", key="topic_search").casefold().strip()
    visible = []
    for theme in themes:
        children = [s for s in theme["subtopics"] if not search or search in
                    (theme["title"] + " " + s["title"] + " " + " ".join(s.get("source_titles", []))).casefold()]
        if children:
            visible.append((theme, children))
    if not visible:
        st.info("Nenhum assunto encontrado. Tente outra palavra.")
        return
    st.caption("Abra um tema para consultar seus assuntos e páginas de origem.")
    columns = st.columns(2, gap="large")
    for index, (theme, children) in enumerate(visible):
        with columns[index % 2]:
            with st.expander(f"{theme['title']} · {len(children)} assuntos", expanded=bool(search)):
                for sub in children:
                    with st.container(border=True):
                        st.markdown("**" + sub["title"] + "**")
                        st.caption("Páginas do PDF: " + _pages(sub["pages"]))
                        sources = sub.get("source_titles", [])
                        if sources:
                            with st.expander("Seções de origem"):
                                for title in sources:
                                    st.write(title)
    st.caption("Este mapa organiza assuntos; não confirma exigências nem substitui a leitura das páginas indicadas.")


def main():
    st.title("🧭 LicitGraphAi")
    st.write("Um mapa organizado para entender o que seu edital aborda.")
    stored = st.session_state.get("result")
    if stored and stored.get("map_version") != 3:
        _reset()
        st.info("Envie novamente o PDF para organizar os assuntos com IA.")
    with st.sidebar:
        st.header("Seu documento")
        with st.expander("Conexão com a API"):
            api_url = st.text_input("Endereço", value=API_URL).rstrip("/")
        try:
            response = requests.get(f"{api_url}/status", timeout=3)
            status = response.json()
            if not status.get("ready", True) or status.get("analysis_mode") == "demo":
                st.warning("Configure a chave Groq e desative o modo demonstração para organizar os assuntos.")
        except (requests.RequestException, ValueError):
            st.warning("API indisponível. Inicie o backend para enviar um PDF.")
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


if __name__ == "__main__":
    main()
