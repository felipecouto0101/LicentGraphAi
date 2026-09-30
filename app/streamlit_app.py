"""Mapa navegável de editais: explicações são geradas somente sob demanda."""

import time

import requests
import streamlit as st

st.set_page_config(page_title="LicitGraphAi · Mapa do edital", page_icon="🧭", layout="wide")
API_URL = "http://127.0.0.1:8002"


def _api_error(response):
    try:
        return response.json().get("detail", response.text)
    except ValueError:
        return response.text or f"HTTP {response.status_code}"


def _ask(api_url: str, job_id: str, question: str, topic_id: str | None):
    # Mensagens antigas são apenas contexto; o backend busca novamente as fontes.
    history = [{"role": row["role"], "content": row["content"]}
               for row in st.session_state.get("conversation", [])[-4:]]
    st.session_state.setdefault("conversation", []).append(
        {"role": "user", "content": question, "topic_id": topic_id})
    with st.spinner("Consultando as cláusulas e conferindo as páginas..."):
        try:
            response = requests.post(
                f"{api_url}/job/{job_id}/ask",
                json={"topic_id": topic_id, "question": question, "history": history},
                timeout=360,
            )
            if response.status_code != 200:
                st.error(_api_error(response))
                return
            data = response.json()
            st.session_state["conversation"].append(
                {"role": "assistant", "content": data["answer"],
                 "citations": data.get("citations", []), "topic_id": topic_id})
        except requests.RequestException as exc:
            st.error(f"Falha de conexão com a API: {exc}")


def _prepare(api_url: str, file):
    st.session_state.pop("result", None)
    st.session_state.pop("conversation", None)
    st.session_state.pop("selected_topic", None)
    try:
        response = requests.post(
            f"{api_url}/analyze/upload",
            files={"file": (file.name, file.getvalue(), "application/pdf")},
            timeout=40,
        )
        if response.status_code == 200:
            st.session_state["job_id"] = response.json()["job_id"]
            st.rerun()
        st.error(_api_error(response))
    except requests.RequestException as exc:
        st.error(f"Falha no envio do PDF: {exc}")


def _poll(api_url: str, job_id: str):
    try:
        response = requests.get(f"{api_url}/job/{job_id}", timeout=8)
        if response.status_code != 200:
            st.error(_api_error(response))
            return
        data = response.json()
    except requests.RequestException as exc:
        st.warning(f"Aguardando API: {exc}")
        time.sleep(3)
        st.rerun()
        return
    if data["status"] == "done":
        st.session_state["result"] = data["result"]
        st.rerun()
    if data["status"] == "error":
        st.error(data.get("error") or "Não foi possível preparar o mapa.")
        if st.button("Tentar outro arquivo"):
            st.session_state.pop("job_id", None)
            st.rerun()
        return
    st.info(data.get("progress", {}).get("stage", "Preparando o mapa..."))
    st.caption("As explicações serão geradas apenas quando você abrir um assunto ou fizer uma pergunta.")
    time.sleep(4)
    st.rerun()


def _topic_browser(api_url: str, job_id: str, result: dict):
    themes = result.get("topic_map") or []
    subtopics = {sub["id"]: (theme, sub) for theme in themes
                 for sub in theme.get("subtopics", [])}
    st.caption(f"{len(themes)} temas · {len(subtopics)} subtemas · "
               f"{result.get('chunks_count', 0)} trechos indexados")
    st.info("O mapa usa títulos e trechos do PDF. Confira as páginas: títulos automáticos podem não refletir toda a cláusula.")
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.subheader("Mapa de assuntos")
        search = st.text_input("Encontrar um assunto", placeholder="Ex.: habilitação, prazo, proposta")
        visible = 0
        for theme in themes:
            children = [sub for sub in theme["subtopics"]
                        if not search or search.casefold() in (theme["title"] + " " + sub["title"]).casefold()]
            if not children:
                continue
            visible += len(children)
            with st.expander(f"{theme.get('display_title', theme['title'])} · {len(children)}", expanded=bool(search)):
                if theme.get("display_title") and theme["display_title"] != theme["title"]:
                    st.caption(f"Título no PDF: {theme['title']}")
                if st.button("Organizar nomes com IA", key=f"rename-{theme['id']}"):
                    with st.spinner("Organizando os nomes deste tema..."):
                        try:
                            renamed = requests.post(f"{api_url}/job/{job_id}/organize",
                                                    json={"theme_id": theme["id"]}, timeout=360)
                            if renamed.status_code == 200:
                                st.session_state["result"]["topic_map"] = renamed.json()["topic_map"]
                                st.rerun()
                            st.error(_api_error(renamed))
                        except requests.RequestException as exc:
                            st.error(f"Falha ao organizar nomes: {exc}")
                for sub in children:
                    pages = ", ".join(str(p) for p in sub["pages"][:5])
                    label = sub.get("display_title", sub["title"])
                    if st.button(label, key=f"open-{sub['id']}", use_container_width=True):
                        st.session_state["selected_topic"] = sub["id"]
                        st.rerun()
                    st.caption(f"Páginas {pages}" + ("…" if len(sub["pages"]) > 5 else ""))
        if not visible:
            st.warning("Nenhum assunto corresponde a essa busca.")
    with right:
        selected = st.session_state.get("selected_topic")
        if selected and selected not in subtopics:
            st.session_state.pop("selected_topic", None)
            selected = None
        if selected:
            theme, sub = subtopics[selected]
            st.caption(theme.get("display_title", theme["title"]))
            st.header(sub.get("display_title", sub["title"]))
            if sub.get("display_title") and sub["display_title"] != sub["title"]:
                st.caption(f"Título no PDF: {sub['title']}")
            st.write("Páginas do PDF: " + ", ".join(map(str, sub["pages"])))
            st.caption(f"{len(sub['chunk_ids'])} trechos relacionados. A explicação ainda não foi gerada.")
            if st.button("✨ Explicar este assunto", type="primary", key=f"explain-{selected}"):
                _ask(api_url, job_id, f"Explique o assunto {sub['title']}, "
                     "suas condições, exceções e consequências para o licitante.", selected)
        else:
            st.header("Explore o edital")
            st.write("Escolha um subtema no mapa ou faça uma pergunta sobre o documento.")
        st.divider()
        st.subheader("Conversa sobre o edital")
        for row in st.session_state.get("conversation", []):
            with st.chat_message(row["role"]):
                st.write(row["content"])
                for citation in row.get("citations", []):
                    st.caption(f"Página {citation['page']}: {citation['quote']}")
        question = st.chat_input("Pergunte sobre este assunto ou sobre todo o edital")
        if question:
            # Uma pergunta livre busca em todo o PDF; depois de clicar em um
            # subtema, a explicação do botão fica restrita às suas fontes.
            _ask(api_url, job_id, question, None)
            st.rerun()


def main():
    st.title("🧭 LicitGraphAi")
    st.write("Navegue pelos assuntos do edital. Peça explicações apenas sobre o que interessa a você.")
    api_url = st.sidebar.text_input("URL da API", value=API_URL).rstrip("/")
    try:
        status = requests.get(f"{api_url}/status", timeout=3).json()
        if not status.get("ready", True):
            st.sidebar.caption("A chave Groq será necessária para gerar explicações no chat.")
        if status.get("analysis_mode") == "demo":
            st.sidebar.warning("Modo demonstração ativado. O chat exige uma chave Groq real.")
    except requests.RequestException:
        st.sidebar.warning("API indisponível no momento.")
    file = st.file_uploader("Escolha um edital em PDF", type="pdf")
    if file and st.button("Criar mapa de assuntos", type="primary"):
        _prepare(api_url, file)
    if st.session_state.get("result") and st.session_state.get("job_id"):
        _topic_browser(api_url, st.session_state["job_id"], st.session_state["result"])
    elif st.session_state.get("job_id"):
        _poll(api_url, st.session_state["job_id"])
    else:
        st.caption("O PDF será lido e indexado antes de mostrar o mapa. Nenhuma explicação é gerada nessa etapa.")


if __name__ == "__main__":
    main()
