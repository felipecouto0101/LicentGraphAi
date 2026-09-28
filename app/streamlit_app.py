"""
Interface Streamlit para LicitGraphAi

Frontend para upload de PDFs e visualização de resultados.
"""

import streamlit as st
import requests
import json
from pathlib import Path

# Configuração da página
st.set_page_config(
    page_title="LicitGraphAi - Analisador de Editais",
    page_icon="📄",
    layout="wide"
)

# URL da API FastAPI
API_URL = "http://127.0.0.1:8002"


def _render_job_progress(job_id: str, api_url: str):
    """Faz polling do job e renderiza o progresso em tempo real."""
    import time as _time

    try:
        r = requests.get(f"{api_url}/job/{job_id}", timeout=5)
        if r.status_code != 200:
            st.error("Erro ao consultar progresso.")
            return
        job = r.json()
    except Exception as e:
        st.warning(f"Aguardando resposta da API... ({e})")
        _time.sleep(3)
        st.rerun()
        return

    status = job.get("status", "")
    prog   = job.get("progress", {})

    if status == "done":
        # Análise concluída — armazena resultado e rerenderiza
        st.session_state["result"] = {
            "result": job.get("result", {}),
            "analysis_mode": job.get("analysis_mode", "real"),
        }
        st.session_state.pop("job_id", None)
        st.rerun()
        return

    if status == "error":
        st.error(f"❌ Erro na análise: {job.get('error', 'desconhecido')}")
        st.session_state.pop("job_id", None)
        return

    # Status running / queued — mostra progresso
    stage       = prog.get("stage", "Processando...")
    query_atual = prog.get("query_atual", "")
    query_num   = prog.get("query_num", 0)
    query_total = prog.get("query_total", 0)
    batch_atual = prog.get("batch_atual", 0)
    batch_total = prog.get("batch_total", 0)
    itens       = prog.get("itens_encontrados", 0)

    st.info(f"⏳ **{stage}**")

    # O callback é emitido antes da chamada à LLM. O lote atual ainda não
    # terminou, portanto não deve contar como concluído na barra.
    if stage in {"Analisando edital com IA", "Explicando todos os requisitos"} and batch_total > 0:
        completed = max(0, min(batch_atual - 1, batch_total))
        label = "Extração" if stage == "Analisando edital com IA" else "Explicações"
        st.progress(
            completed / batch_total,
            text=f"{label}: {completed}/{batch_total} lotes concluídos",
        )
        st.caption(f"Em andamento: lote {batch_atual}/{batch_total}" + (f" — {query_atual}" if query_atual else ""))
    elif stage == "Redigindo resumo e requisitos":
        st.caption("Redigindo a síntese; esta etapa ainda não tem porcentagem disponível.")

    if itens > 0:
        st.caption(f"✅ {itens} itens encontrados até agora")

    st.caption("A análise continua em background. Esta página atualiza automaticamente a cada 5s.")

    # Auto-refresh a cada 5 segundos
    _time.sleep(5)
    st.rerun()


def main():
    """Função principal do Streamlit."""
    
    # Header
    st.title("📄 LicitGraphAi - Analisador de Editais")
    st.markdown("Sistema inteligente para análise automática de editais de licitação pública")
    
    # Sidebar
    st.sidebar.title("Configurações")
    api_url = st.sidebar.text_input("URL da API", value=API_URL)

    try:
        api_status = requests.get(f"{api_url}/status", timeout=3).json()
        if not api_status.get("ready", True):
            st.error(api_status.get("configuration_error", "Análise indisponível."))
        if api_status.get("analysis_mode") == "demo":
            st.warning(
                "⚠️ Modo de demonstração: resultados simulados/heurísticos, "
                "sem análise pela Groq. Não use estes resultados para avaliar um edital real."
            )
    except requests.RequestException:
        pass
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Análise", "Status", "Sobre"])
    
    with tab1:
        st.header("📤 Upload e Análise de Edital")
        
        # Upload do PDF
        uploaded_file = st.file_uploader(
            "Selecione o arquivo PDF do edital",
            type="pdf",
            help="Selecione um arquivo PDF contendo o edital da licitação"
        )
        
        # Perfil da empresa (opcional)
        st.subheader("Perfil da Empresa (Opcional)")
        company_name = st.text_input("Nome da Empresa")
        company_cnpj = st.text_input("CNPJ")
        
        company_profile = None
        if company_name or company_cnpj:
            company_profile = {
                "name": company_name,
                "cnpj": company_cnpj
            }
        
        # Botão de análise
        if uploaded_file is not None:
            st.info(f"Arquivo selecionado: {uploaded_file.name}")

            if st.button("🚀 Iniciar Análise", type="primary"):
                try:
                    files = {"file": uploaded_file}
                    data = {}
                    if company_profile:
                        data["company_profile"] = json.dumps(company_profile)

                    response = requests.post(
                        f"{api_url}/analyze/upload",
                        files=files,
                        data=data,
                        timeout=30,
                    )

                    if response.status_code == 200:
                        job_data = response.json()
                        st.session_state["job_id"] = job_data["job_id"]
                        st.session_state["api_url"] = api_url
                        st.session_state["result"] = None
                        st.rerun()
                    else:
                        st.error(f"❌ Erro ao iniciar análise: {response.text}")

                except requests.exceptions.ConnectionError:
                    st.error("❌ Não foi possível conectar à API.")
                except Exception as e:
                    st.error(f"❌ Erro: {str(e)}")

        # Polling de progresso enquanto job estiver rodando
        if st.session_state.get("job_id") and not st.session_state.get("result"):
            job_id  = st.session_state["job_id"]
            api_url_job = st.session_state.get("api_url", api_url)
            _render_job_progress(job_id, api_url_job)

        # Exibe resultado quando pronto
        if st.session_state.get("result"):
            st.success("✅ Análise concluída com sucesso!")
            display_results(st.session_state["result"])
            if st.button("🔄 Nova Análise"):
                st.session_state.pop("job_id", None)
                st.session_state.pop("result", None)
                st.rerun()

        if not uploaded_file and not st.session_state.get("job_id") and not st.session_state.get("result"):
            st.info("👆 Selecione um arquivo PDF para começar")
    
    with tab2:
        st.header("📊 Status do Sistema")

        if st.button("🔄 Atualizar Status"):
            st.rerun()

        try:
            response = requests.get(f"{api_url}/status", timeout=5)
            if response.status_code == 200:
                status = response.json()

                col1, col2, col3 = st.columns(3)
                col1.metric("Status", status["status"])
                col2.metric("Versão", status["version"])
                col3.metric("Orquestração", status["orchestration"])

                st.subheader("Nós Disponíveis")
                for node in status["nodes"]:
                    st.success(f"✅ {node}")
            else:
                st.error("❌ Não foi possível obter o status do sistema")

        except requests.exceptions.ConnectionError:
            st.warning("⚠️ API não está respondendo em `http://127.0.0.1:8002`.")
            st.info(
                "A API pode estar ainda inicializando. "
                "Clique em **Atualizar Status** após alguns instantes."
            )
            st.code("python start_app.py", language="bash")
        except requests.exceptions.Timeout:
            st.warning("⚠️ API demorou para responder. Clique em Atualizar Status.")
        except Exception as e:
            st.error(f"❌ Erro: {str(e)}")
    
    with tab3:
        st.header("ℹ️ Sobre o LicitGraphAi")
        
        st.markdown("""
        ### Sistema de Análise de Editais
        
        O LicitGraphAi é um sistema inteligente para análise automática de editais de licitação pública (Nova Lei 14.133).
        
        ### Funcionalidades
        
        - 📄 **Leitura e Fragmentação**: Extrai texto do PDF e fragmenta em partes menores
        - 🔍 **Embeddings**: Gera representações vetoriais do texto
        - 🤖 **Análise com IA**: Analisa requisitos usando inteligência artificial
        - 📋 **Checklist**: Gera checklist de documentos necessários
        
        ### Stack Tecnológica
        
        - Python 3.11+
        - LangChain + LangGraph
        - Groq API + OpenAI GPT-OSS-120b
        - sentence-transformers
        - ChromaDB
        - FastAPI + Streamlit
        
        ### Como Usar
        
        1. Certifique-se de que a API FastAPI está rodando:
           ```bash
           python -m uvicorn app.api.main:app --reload --host 127.0.0.1 --port 8002
           ```
        
        2. Execute o Streamlit:
           ```bash
           streamlit run app/streamlit_app.py
           ```
        
        3. Faça upload do PDF do edital
        4. Clique em "Iniciar Análise"
        5. Aguarde o processamento
        6. Visualize os resultados
        """)


def display_results(result):
    """Exibe os resultados da análise de forma legível."""

    st.header("📊 Resultados da Análise")
    if result.get("analysis_mode") == "demo":
        st.warning("⚠️ Resultado de demonstração, sem análise pela Groq.")

    result_data = result.get("result", {})

    result_tab1, result_tab2, result_tab3, result_tab4, result_tab5, result_tab6 = st.tabs([
        "📈 Resumo",
        "❓ Perguntas Respondidas",
        "🔍 Análise de Requisitos",
        "📋 Checklist de Documentos",
        "📁 Execução e anexos",
        "🗂 JSON Completo",
    ])

    analysis      = result_data.get("analysis") or {}
    structured    = analysis.get("structured_analysis", {})
    critical      = analysis.get("critical_analysis", {})
    llm_analysis  = analysis.get("llm_analysis")  # None em mock mode
    narrative     = analysis.get("explanatory_report") or {}
    narrative_error = analysis.get("explanatory_error")
    rag_sources = analysis.get("rag_sources") or {}
    checklist_data = result_data.get("checklist") or {}
    resumo        = checklist_data.get("resumo", {})
    checklist     = checklist_data.get("checklist", {})

    # ── helpers locais ────────────────────────────────────────────
    def dedup_ordered(lst):
        seen, out = set(), []
        for x in lst:
            x = x.strip()
            if x and x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def aggregate(field):
        items = []
        for data in structured.values():
            items.extend(data.get(field, []))
        return dedup_ordered(items)

    def aggregate_critical(field):
        items = []
        for data in critical.values():
            items.extend(data.get(field, []))
        return dedup_ordered(items)

    all_tech      = aggregate("technical_requirements")
    selection_items = analysis.get("selection_process")
    detailed_explanations = analysis.get("detailed_explanations") or {}
    all_docs_raw  = aggregate("documentation")
    all_deadlines = aggregate("deadlines")
    all_risks     = aggregate_critical("critical_points")
    all_recs      = dedup_ordered(aggregate_critical("recommendations"))

    # ──────────────────────────────────────────────────────────────
    # TAB 1 — Resumo executivo
    # ──────────────────────────────────────────────────────────────
    with result_tab1:
        if narrative.get("resumo"):
            st.subheader("Visão geral do edital")
            for paragraph in narrative["resumo"].split("\n\n"):
                if paragraph.strip():
                    st.write(paragraph.strip())
            coverage = narrative.get("coverage", {})
            if any(v.get("included", 0) < v.get("total", 0) for v in coverage.values()):
                st.caption(
                    "Síntese baseada em uma amostra dos itens extraídos em cada tema. "
                    "Confira os detalhes e o PDF original antes de tomar decisões."
                )
            st.divider()
        elif narrative_error:
            st.warning(
                "Não foi possível redigir o texto explicativo. "
                "Os itens extraídos estão disponíveis nas demais abas."
            )
        if rag_sources:
            with st.expander("📖 Trechos do PDF para conferência", expanded=False):
                st.caption(
                    "Páginas físicas do arquivo PDF. Estes trechos ajudam a conferir "
                    "o relatório; revise o edital original antes de confiar em uma conclusão."
                )
                for topic, sources in rag_sources.items():
                    if sources:
                        st.markdown(f"**{topic.replace('_', ' ').capitalize()}**")
                        for source in sources:
                            page = source.get("page")
                            label = f"Página {page}" if page is not None else "Página desconhecida"
                            st.write(f"{label}, trecho {source.get('chunk_id', '?')}: {source.get('excerpt', '')}")
        chunks_count     = result_data.get("chunks_count", 0)
        embeddings_count = (result_data.get("embeddings") or {}).get("total_embeddings", 0)
        docs_count       = resumo.get("total_documentos", 0)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Trechos extraídos",   chunks_count)
        col2.metric("Embeddings gerados",  embeddings_count)
        col3.metric("Documentos no checklist", docs_count)
        col4.metric("Seções analisadas",   analysis.get("total_chunks_processed", 0))

        # Nível de risco
        st.divider()
        st.subheader("Avaliação de Risco")

        if critical:
            risk_counts = {"ALTO": 0, "MEDIO": 0, "BAIXO": 0}
            for d in critical.values():
                risk_counts[d.get("risk_level", "BAIXO")] += 1

            overall = "ALTO" if risk_counts["ALTO"] else ("MEDIO" if risk_counts["MEDIO"] else "BAIXO")
            icon    = {"ALTO": "🔴", "MEDIO": "🟡", "BAIXO": "🟢"}[overall]
            st.markdown(f"### {icon} Risco Geral: {overall}")

            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("🔴 Alto",  risk_counts["ALTO"])
            rc2.metric("🟡 Médio", risk_counts["MEDIO"])
            rc3.metric("🟢 Baixo", risk_counts["BAIXO"])
        else:
            st.info("Avaliação de risco não disponível.")

        # Prazos — versão resumida no Resumo (máx 5)
        if all_deadlines:
            st.divider()
            st.subheader("⏰ Principais Prazos")
            for d in all_deadlines[:5]:
                st.write(f"• {d}")
            if len(all_deadlines) > 5:
                st.caption(f"+ {len(all_deadlines) - 5} prazos adicionais na aba Análise de Requisitos.")

        # Recomendações prioritárias
        if all_recs:
            st.divider()
            st.subheader("💡 Recomendações")
            for rec in all_recs[:4]:
                st.success(f"✔ {rec}")

        # Objeto do edital (só disponível com LLM)
        if llm_analysis and llm_analysis.get("objeto"):
            st.divider()
            st.subheader("📌 Objeto do Edital")
            st.info(llm_analysis["objeto"])

    # ──────────────────────────────────────────────────────────────
    # TAB 2 — Perguntas e Respostas RAG
    # ──────────────────────────────────────────────────────────────
    with result_tab2:
        rag_answers = analysis.get("rag_answers", {})

        if not rag_answers:
            st.info(
                "ℹ️ As respostas por pergunta estão disponíveis apenas com IA real. "
                "Configure `NODE3_MOCK_MODE=false` e `GROQ_API_KEY` no `.env`."
            )
        else:
            risk_icon = {"ALTO": "🔴", "MEDIO": "🟡", "BAIXO": "🟢"}

            for key, label, _ in analysis.get("_rag_queries_meta", []) or [
                # fallback: percorre na ordem original se não vier metadado
                (k, v["label"], None) for k, v in rag_answers.items()
            ]:
                ans = rag_answers.get(key, {})
                if not ans:
                    continue

                risco = ans.get("nivel_risco", "BAIXO")
                icon  = risk_icon.get(risco, "⚪")
                label_display = ans.get("label", label)

                with st.expander(f"{icon} {label_display}", expanded=False):
                    resposta = ans.get("resposta", "")
                    if resposta:
                        st.markdown(f"**{resposta}**")

                    detalhes = ans.get("detalhes", [])
                    if detalhes:
                        with st.expander(f"Ver {len(detalhes)} itens extraídos", expanded=False):
                            for item in detalhes:
                                st.write(f"• {item}")

                    sources = rag_sources.get(key, [])
                    if sources:
                        with st.expander("Trechos do PDF relacionados", expanded=False):
                            for source in sources:
                                page = source.get("page")
                                label = f"Página {page}" if page is not None else "Página desconhecida"
                                st.write(f"{label}: {source.get('excerpt', '')}")

                    obs = ans.get("observacao", "")
                    if obs:
                        st.caption(f"ℹ️ {obs}")

    # ──────────────────────────────────────────────────────────────
    # TAB 3 — Análise de Requisitos
    # ──────────────────────────────────────────────────────────────
    with result_tab3:
        if narrative.get("requisitos"):
            st.subheader("Explicação dos requisitos")
            for paragraph in narrative["requisitos"].split("\n\n"):
                if paragraph.strip():
                    st.write(paragraph.strip())
            coverage = narrative.get("coverage", {})
            if any(v.get("included", 0) < v.get("total", 0) for v in coverage.values()):
                st.caption("Texto baseado em amostra; consulte todos os itens abaixo e o edital.")
        elif narrative_error:
            st.warning("Texto explicativo indisponível; consulte os itens extraídos abaixo.")

        if not structured:
            st.info("Análise de requisitos não disponível.")
        else:
            st.caption(
                "As explicações cobrem os itens identificados na análise. "
                "Confira as cláusulas no PDF: a extração automática pode deixar informações de fora."
            )

            def render_explained_items(topic, items):
                explained = detailed_explanations.get(topic, [])
                complete = (
                    len(explained) == len(items)
                    and all(row.get("id") == i and row.get("item") == item
                            and row.get("explicacao")
                            for i, (row, item) in enumerate(zip(explained, items), 1))
                )
                if not complete:
                    st.caption("Explicações item a item indisponíveis nesta análise; itens originais abaixo.")
                for i, item in enumerate(items, 1):
                    st.markdown(f"**{i}. {item}**")
                    if complete:
                        detail = explained[i - 1]
                        status = detail.get("situacao")
                        if status == "incerta":
                            st.warning("Interpretação pendente: a explicação abaixo é preliminar e a aplicação da regra não foi confirmada.")
                        elif status == "nao_aplicavel":
                            st.info("Regra alternativa: a condição descrita não corresponde ao modo de disputa indicado no início deste PDF.")
                        elif status == "condicional":
                            st.info("Regra condicional: depende da situação descrita na cláusula.")
                        if detail.get("condicao"):
                            st.caption(f"Condição no PDF: {detail['condicao']}")
                        if detail.get("explicacao_preliminar"):
                            st.caption("Entendendo o tópico (leitura preliminar)")
                        st.write(detail["explicacao"])
                        evidence = detail.get("evidencia")
                        if evidence:
                            page = evidence.get("pagina")
                            label = f"Página {page}" if page is not None else "Página não informada"
                            st.caption(f"Trecho relacionado — {label}: {evidence.get('trecho', '')}")
                        elif status == "incerta":
                            pages = detail.get("paginas_para_revisao") or []
                            if pages:
                                labels = ", ".join(str(page) for page in pages)
                                st.caption(f"Páginas sugeridas para revisão: {labels}. Busca por palavras; a cláusula não foi confirmada.")
                            else:
                                st.caption("Página não localizada automaticamente para este tópico.")

            # Condições de participação e habilitação (não são etapas da disputa).
            participation_items = all_tech if selection_items is not None or not llm_analysis else []
            if participation_items:
                with st.expander(f"📋 Condições de participação e habilitação — {len(participation_items)} item(ns)", expanded=False):
                    render_explained_items("participacao", participation_items)
            elif selection_items is not None:
                st.info("Nenhuma condição de participação ou habilitação identificada.")

            # A saída antiga colocava seleção indevidamente em technical_requirements.
            # Para análises antigas, mantenha a lista visível com o título correto.
            selection_display = selection_items if selection_items is not None else (all_tech if llm_analysis else [])
            if selection_display:
                with st.expander(f"⚖️ Disputa, julgamento e avaliação — {len(selection_display)} item(ns)", expanded=False):
                    render_explained_items("selecao", selection_display)

            # Prazos — todos aqui
            if all_deadlines:
                with st.expander(f"⏰ Prazos e Cronogramas — {len(all_deadlines)} item(ns)", expanded=False):
                    for d in all_deadlines:
                        st.write(f"• {d}")
            else:
                st.info("Nenhum prazo identificado.")

            # Pontos críticos — colapsado por padrão, todos os válidos
            if all_risks:
                with st.expander(
                    f"⚠️ Pontos Críticos — {len(all_risks)} identificados",
                    expanded=False,
                ):
                    for i, risk in enumerate(all_risks, 1):
                        st.warning(f"**{i}.** {risk}")
            
            # Todas as recomendações
            if all_recs:
                st.divider()
                with st.expander(f"💡 Recomendações extraídas — {len(all_recs)}", expanded=False):
                    for rec in all_recs:
                        st.write(f"• {rec}")

    # ──────────────────────────────────────────────────────────────
    # TAB 4 — Checklist de Documentos
    # ──────────────────────────────────────────────────────────────
    with result_tab4:
        if not checklist:
            st.info("Checklist não disponível. Nenhum documento foi identificado no edital.")
        else:
            st.caption(
                "Documentos identificados automaticamente. A apresentação, a etapa e a "
                "obrigatoriedade de cada um precisam ser confirmadas no edital."
            )
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Documentos citados", resumo.get("total_documentos", 0))
            col2.metric("Obrigatórios confirmados", resumo.get("obrigatorios", 0))
            col3.metric("Opcionais confirmados", resumo.get("opcionais", 0))
            col4.metric("A confirmar", resumo.get("a_confirmar", 0))
            col5.metric("Categorias", resumo.get("categorias", 0))

            st.divider()

            category_labels = {
                "habilitacao": ("🪪", "Cadastro e identificação"),
                "fiscal":      ("🧾", "Regularidade Fiscal"),
                "tecnica":     ("⚙️", "Qualificação Técnica"),
                "juridica":    ("⚖️", "Documentação Jurídica"),
                "trabalhista": ("👷", "Regularidade Trabalhista"),
                "economica":   ("📊", "Qualificação Econômica"),
                "proposta":    ("📄", "Proposta e planilhas"),
                "outros":      ("📂", "Outros documentos a conferir"),
            }

            for category, data in checklist.items():
                icon, label = category_labels.get(category, ("📁", category.upper()))
                itens = data.get("itens", [])
                if not itens:
                    continue

                with st.expander(f"{icon} {label} — {len(itens)} documento(s)", expanded=True):
                    # Cabeçalho da tabela
                    hc = st.columns([0.04, 0.52, 0.22, 0.22])
                    hc[1].markdown("**Documento**")
                    hc[2].markdown("**Tipo**")
                    hc[3].markdown("**Prazo**")
                    st.markdown("---")

                    for item in itens:
                        doc_name    = item.get("documento", "—").title()
                        obrigatorio = item.get("obrigatorio")
                        status      = item.get("status", "pendente")
                        prazo       = item.get("prazo")

                        badge = (
                            "🔴 Obrigatório" if obrigatorio is True
                            else "🟡 Opcional" if obrigatorio is False
                            else "⚪ A confirmar"
                        )
                        status_icon = {"pendente": "⬜", "ok": "✅", "faltando": "❌"}.get(status, "⬜")

                        cols = st.columns([0.04, 0.52, 0.22, 0.22])
                        cols[0].write(status_icon)
                        cols[1].write(doc_name)
                        cols[2].write(badge)
                        cols[3].write(prazo if prazo else "—")

    # ──────────────────────────────────────────────────────────────
    # TAB 5 — Documentos produzidos depois da contratação e anexos do órgão
    # ──────────────────────────────────────────────────────────────
    with result_tab5:
        sections = (
            ("Atividades e entregas previstas", analysis.get("execution_items") or []),
            ("Documentos da etapa de execução", analysis.get("documentos_execucao") or []),
            ("Anexos fornecidos para consulta", analysis.get("anexos_referencia") or []),
            ("Documentos com etapa ou exigência incerta", analysis.get("pendencias_documentais") or []),
        )
        st.caption("Confira no PDF a etapa e as condições de cada item; os itens incertos não entram no checklist.")
        if not any(items for _, items in sections):
            st.info("Nenhum item separado nesta análise. Para relatórios antigos, rode uma nova análise.")
        for label, items in sections:
            if items:
                with st.expander(f"{label} — {len(items)} item(ns)", expanded=False):
                    for item in items:
                        st.write(f"• {item}")

    # ──────────────────────────────────────────────────────────────
    # TAB 6 — JSON (debug)
    # ──────────────────────────────────────────────────────────────
    with result_tab6:
        st.caption("Dados brutos retornados pela API — útil para debug.")
        st.json(result)


if __name__ == "__main__":
    main()
