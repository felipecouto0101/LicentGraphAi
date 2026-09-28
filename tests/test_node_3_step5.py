"""
Testes para Etapa 5 do Nó 3: Integração completa com Nó 2

TDD Approach: Testes escritos antes da implementação
"""

import pytest
from typing import Dict, List


class TestTransientGroqFailures:
    def test_minute_limit_reports_real_cause_without_rotating_keys(self, monkeypatch):
        from types import SimpleNamespace
        from app.rag import node_3_analyzer as module

        class MinuteLimit(Exception):
            status_code = 429

        analyzer = object.__new__(module.Node3RequirementAnalyzer)
        analyzer._api_keys = ["one", "two", "three"]
        analyzer._current_key_idx = 0
        calls, waits = [], []

        def invoke(messages):
            calls.append(messages)
            raise MinuteLimit("limit per minute")

        analyzer.llm = SimpleNamespace(invoke=invoke)
        monkeypatch.setattr(module.time, "sleep", waits.append)
        with pytest.raises(RuntimeError, match="limite de requisições por minuto"):
            analyzer._invoke_with_rotation(["message"])

        assert len(calls) == 4
        assert waits == [60, 60, 60]
        assert analyzer._current_key_idx == 0

    def test_503_recovers_without_changing_key(self, monkeypatch):
        from types import SimpleNamespace
        from app.rag import node_3_analyzer as module

        class ServerUnavailable(Exception):
            status_code = 503

        analyzer = object.__new__(module.Node3RequirementAnalyzer)
        analyzer._api_keys = ["key"]
        analyzer._current_key_idx = 0
        calls = []
        waits = []

        def invoke(messages):
            calls.append(messages)
            if len(calls) <= 2:
                raise ServerUnavailable("upstream connect error")
            return SimpleNamespace(content="ok")

        analyzer.llm = SimpleNamespace(invoke=invoke)
        monkeypatch.setattr(module.time, "sleep", waits.append)
        response = analyzer._invoke_with_rotation(["message"])

        assert response.content == "ok"
        assert len(calls) == 3
        assert waits == [5, 15]
        assert analyzer._current_key_idx == 0

    def test_persistent_503_stops_after_bounded_retries(self, monkeypatch):
        from types import SimpleNamespace
        from app.rag import node_3_analyzer as module

        class ServerUnavailable(Exception):
            status_code = 503

        analyzer = object.__new__(module.Node3RequirementAnalyzer)
        analyzer._api_keys = ["key"]
        calls = []
        waits = []

        def invoke(messages):
            calls.append(messages)
            raise ServerUnavailable("upstream connect error")

        analyzer.llm = SimpleNamespace(invoke=invoke)
        monkeypatch.setattr(module.time, "sleep", waits.append)
        with pytest.raises(ServerUnavailable):
            analyzer._invoke_with_rotation(["message"])

        assert len(calls) == 4
        assert waits == [5, 15, 30]

    def test_other_client_errors_are_not_retried(self, monkeypatch):
        from types import SimpleNamespace
        from app.rag import node_3_analyzer as module

        class BadRequest(Exception):
            status_code = 400

        analyzer = object.__new__(module.Node3RequirementAnalyzer)
        analyzer._api_keys = ["key"]
        analyzer.llm = SimpleNamespace(invoke=lambda messages: (_ for _ in ()).throw(BadRequest("bad request")))
        monkeypatch.setattr(module.time, "sleep", lambda seconds: pytest.fail("unexpected retry"))
        with pytest.raises(BadRequest):
            analyzer._invoke_with_rotation(["message"])


class TestExplanationGrounding:
    def test_pending_item_keeps_candidate_page_without_claiming_evidence(self, monkeypatch):
        import json
        from types import SimpleNamespace
        from app.rag import node_3_analyzer as module

        node = object.__new__(module.Node3RequirementAnalyzer)
        agg = {"requisitos_participacao": ["Registro no CREA"], "selecao": []}
        chunks = [{"content": "Registro profissional no CREA é exigido.",
                   "metadata": {"page": 18}}]

        def invoke(messages):
            entry = json.loads(messages[-1].content.split("ITENS: ", 1)[1])[0]
            return SimpleNamespace(content=json.dumps({"explicacoes": [{
                "id": entry["id"], "situacao": "incerta", "evidencia": None,
                "explicacao": "O registro profissional precisa ser verificado antes da participação no processo.",
                "explicacao_basica": "O tópico trata do registro profissional no CREA.",
            }]}))

        monkeypatch.setattr(node, "_invoke_with_rotation", invoke)
        result = node._explain_all_requirements(agg, chunks=chunks)["participacao"][0]
        assert result["paginas_para_revisao"] == [18]
        assert result["evidencia"] is None
        assert result["situacao"] == "incerta"

    def test_saved_pending_item_gets_page_without_another_llm_call(self, tmp_path, monkeypatch):
        import json
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer

        node = object.__new__(Node3RequirementAnalyzer)
        node._checkpoint_file = tmp_path / "saved.json"
        monkeypatch.setattr(node, "_invoke_with_rotation", lambda _: pytest.fail("LLM não necessária"))
        agg = {"requisitos_participacao": ["Registro no CREA"], "selecao": []}
        saved = {"explanations": {"participacao": {"1": {
            "id": 1, "item": "Registro no CREA", "situacao": "incerta",
            "explicacao": "A empresa precisa de registro profissional.",
        }}, "selecao": {}}}

        result = node._explain_all_requirements(agg, saved, [{
            "content": "O registro no CREA consta da exigência técnica.",
            "metadata": {"page": 21},
        }])
        assert result["participacao"][0]["paginas_para_revisao"] == [21]
        assert json.loads(node._checkpoint_file.read_text(encoding="utf-8"))[
            "explanations"]["participacao"]["1"]["paginas_para_revisao"] == [21]

    def test_literal_quote_is_required_before_publishing_interpretation(self, monkeypatch):
        import json
        from types import SimpleNamespace
        from app.rag import node_3_analyzer as module

        node = object.__new__(module.Node3RequirementAnalyzer)
        source = {"content": "A plataforma analisa o cadastro em até 24 horas úteis.",
                  "metadata": {"page": 3}}
        agg = {"requisitos_participacao": ["Cadastro na plataforma com antecedência"],
               "selecao": []}

        def reply(evidence, status="aplicavel", hours=24):
            def invoke(messages):
                entries = json.loads(messages[-1].content.split("ITENS: ", 1)[1])
                return SimpleNamespace(content=json.dumps({"explicacoes": [{
                    "id": entries[0]["id"], "situacao": status, "evidencia": evidence,
                    "explicacao": f"O cadastro deve ser antecipado porque sua análise leva até {hours} horas úteis.",
                    "explicacao_basica": "O cadastro precisa ser feito antes da participação.",
                }]}))
            return invoke

        valid = {"pagina": 3, "trecho": "A plataforma analisa o cadastro em até 24 horas úteis."}
        monkeypatch.setattr(node, "_invoke_with_rotation", reply(valid))
        result = node._explain_all_requirements(agg, chunks=[source])
        assert result["participacao"][0]["situacao"] == "aplicavel"
        assert result["participacao"][0]["evidencia"]["pagina"] == 3

        monkeypatch.setattr(node, "_invoke_with_rotation", reply({"pagina": 3, "trecho":
            "A plataforma analisa o cadastro em até 48 horas úteis."}))
        result = node._explain_all_requirements(agg, chunks=[source])
        assert result["participacao"][0]["situacao"] == "incerta"
        assert "24 horas" not in result["participacao"][0]["explicacao"]
        assert result["participacao"][0]["explicacao"] == "O cadastro precisa ser feito antes da participação."
        assert result["participacao"][0]["explicacao_preliminar"] is True

        monkeypatch.setattr(node, "_invoke_with_rotation", reply(valid, hours=48))
        result = node._explain_all_requirements(agg, chunks=[source])
        assert result["participacao"][0]["situacao"] == "incerta"
        assert "48 horas" not in result["participacao"][0]["explicacao"]

    def test_version_three_retains_verified_explanations_and_refills_only_pending(self, tmp_path):
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer

        node = object.__new__(Node3RequirementAnalyzer)
        node.checkpoint_directory = tmp_path
        node.model_name = "modelo-teste"
        chunks = [{"content": "Regra do cadastro", "metadata": {"page": 1}}]
        agg = {"requisitos_participacao": [], "selecao": []}
        saved = node._load_checkpoint(chunks, 1, agg)
        saved["next_batch"] = 1
        saved["explanation_version"] = 3
        saved["explanations"]["participacao"] = {
            "1": {"item": "Comprovar cadastro", "explicacao": "Cadastro explicado."},
            "2": {"item": "Documento seguinte",
                  "explicacao": node.OLD_PENDING_EXPLANATION},
        }
        node._save_checkpoint(saved)

        resumed = node._load_checkpoint(chunks, 1, agg)
        assert resumed["next_batch"] == 1
        assert list(resumed["explanations"]["participacao"]) == ["1"]
        assert resumed["explanation_version"] == node.EXPLANATION_VERSION

    def test_prompt_uses_relevant_source_and_related_items(self, monkeypatch):
        import json
        from types import SimpleNamespace
        from app.rag import node_3_analyzer as module

        node = object.__new__(module.Node3RequirementAnalyzer)
        prompts = []

        def invoke(messages):
            prompt = messages[-1].content
            prompts.append(prompt)
            entries = json.loads(prompt.split("ITENS: ", 1)[1])
            return SimpleNamespace(content=json.dumps({"explicacoes": [
                {"id": row["id"], "explicacao":
                 "A regra exige cadastro correto. Os registros devem refletir os dados da empresa."}
                for row in entries
            ]}))

        monkeypatch.setattr(node, "_invoke_with_rotation", invoke)
        monkeypatch.setattr(module.time, "sleep", lambda _: None)
        agg = {"requisitos_participacao": ["Dados cadastrais exatos e atualizados",
                "Cadastro na plataforma com antecedência"], "selecao": []}
        chunks = [{"content": "Os dados cadastrais na plataforma devem permanecer atualizados.",
                   "metadata": {"page": 4}},
                  {"content": "Cronograma de execução da obra.", "metadata": {"page": 9}}]

        result = node._explain_all_requirements(agg, chunks=chunks)

        assert len(result["participacao"]) == 2
        assert '"pagina": 4' in prompts[0]
        assert '"pagina": 9' not in prompts[0]
        assert "não diga que o edital não lista suas condições" in prompts[0]

    def test_old_checkpoint_keeps_extraction_and_regenerates_explanations(self, tmp_path):
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer

        node = object.__new__(Node3RequirementAnalyzer)
        node.checkpoint_directory = tmp_path
        node.model_name = "modelo-teste"
        chunks = [{"content": "Dados cadastrais atualizados", "section": "cadastro",
                   "metadata": {"page": 1}}]
        agg = {"requisitos_participacao": []}
        old = node._load_checkpoint(chunks, 2, agg)
        old["next_batch"] = 1
        old["agg"]["requisitos_participacao"] = ["Dados cadastrais atualizados"]
        old["explanations"]["participacao"]["1"] = "Texto genérico anterior que precisa ser refeito."
        old.pop("explanation_version")
        node._save_checkpoint(old)

        resumed = node._load_checkpoint(chunks, 2, agg)

        assert resumed["next_batch"] == 1
        assert resumed["agg"]["requisitos_participacao"] == ["Dados cadastrais atualizados"]
        assert resumed["explanations"] == {"participacao": {}, "selecao": {}}
        assert resumed["explanation_version"] == node.EXPLANATION_VERSION


class TestNode3IntegrationWithNode2:
    """Testes para integração do Nó 3 com Nó 2."""
    
    def test_process_chunks_from_node2(self):
        """Testa processamento de chunks vindos do Nó 2."""
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer
        
        node3 = Node3RequirementAnalyzer(api_key="test_key", mock_mode=True)
        
        # Simula chunks do Nó 2
        node2_chunks = [
            {
                "content": "Processador Intel Core i5 ou superior",
                "section": "exigencias_tecnicas",
                "metadata": {"chunk_id": 0}
            },
            {
                "content": "Prazo de entrega: 30 dias",
                "section": "prazos",
                "metadata": {"chunk_id": 1}
            }
        ]
        
        result = node3.process_from_node2(node2_chunks)
        
        assert "total_chunks_processed" in result
        assert "structured_analysis" in result
        assert "critical_analysis" in result
        assert result["total_chunks_processed"] == 2
    
    def test_explanatory_report_is_bounded_and_excludes_document_list(self, monkeypatch):
        from types import SimpleNamespace
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer

        analyzer = object.__new__(Node3RequirementAnalyzer)
        agg = {
            "objeto": "Serviço de manutenção",
            "nivel_risco": "MEDIO",
            "documentos": ["DOCUMENTO_PRIVADO_QUE_NAO_DEVE_ENTRAR"],
            "requisitos_participacao": [f"Critério {i}" for i in range(25)],
            "selecao": ["Menor preço"], "prazos": [], "custos": [],
            "entregas": [], "eliminacao": [], "riscos": [],
        }
        captured = []
        def fake_invoke(messages):
            captured.append(messages[-1].content)
            return SimpleNamespace(content='{"resumo":"Objeto e prazo a verificar.",'
                                           '"requisitos":"A seleção usa menor preço."}')
        monkeypatch.setattr(analyzer, "_invoke_with_rotation", fake_invoke)
        report = analyzer._synthesize_explanatory_report(agg)

        assert report["coverage"]["participacao"] == {"included": 12, "total": 25}
        assert "Critério 0" in captured[0] and "Critério 24" in captured[0]
        assert "DOCUMENTO_PRIVADO_QUE_NAO_DEVE_ENTRAR" not in captured[0]
        assert report["requisitos"] == "A seleção usa menor preço."

    def test_full_chunk_retrieval_and_page_provenance(self, monkeypatch):
        import json
        from types import SimpleNamespace
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer

        analyzer = object.__new__(Node3RequirementAnalyzer)
        chunk = {"content": "Início " + ("x" * 700) + " SENTINELA_FIM",
                 "section": "prazos", "metadata": {"chunk_id": 2, "page": 3}}
        captured = []
        data = {
            "objeto": "", "documentos": ["Atestado técnico", "Minuta de referência"],
            "documentos_execucao": ["ART de execução"],
            "anexos_referencia": ["Minuta de referência", "Projeto Básico"],
            "pendencias_documentais": ["Documento sem etapa"],
            "requisitos_participacao": ["Registro técnico"],
            "prazos": ["30 dias"], "custos": [],
            "selecao": ["Menor preço global", "menor preco global!"], "entregas": [],
            "eliminacao": [], "riscos": [], "nivel_risco": "BAIXO",
        }
        def invoke(messages, *args):
            captured.append(messages[-1].content)
            return SimpleNamespace(content=json.dumps(data))
        monkeypatch.setattr(analyzer, "_invoke_with_rotation", invoke)
        monkeypatch.setattr(analyzer, "_retrieve_rag_chunks",
                            lambda **kwargs: {"prazos": [chunk]})
        monkeypatch.setattr(analyzer, "_synthesize_explanatory_report",
                            lambda agg, refs: {"resumo": "Prazo [p. 3].",
                                               "requisitos": "Sem requisito.",
                                               "coverage": {}})
        monkeypatch.setattr(analyzer, "_explain_all_requirements",
                            lambda agg, checkpoint=None, chunks=None: {"participacao": [], "selecao": []})

        result = analyzer._process_with_rag_llm([chunk])
        assert "SENTINELA_FIM" in captured[0]
        assert result["rag_sources"]["prazos"][0]["page"] == 3
        assert result["explanatory_report"]["resumo"] == "Prazo [p. 3]."
        assert result["selection_process"] == ["Menor preço global"]
        assert result["structured_analysis"][0]["technical_requirements"] == ["Registro técnico"]
        assert result["llm_analysis"]["documentos_exigidos"] == ["Atestado técnico"]
        assert result["documentos_execucao"] == ["ART de execução"]
        assert result["anexos_referencia"] == ["Projeto Básico"]
        assert result["pendencias_documentais"] == ["Documento sem etapa", "Minuta de referência"]

    def test_explains_every_extracted_item_across_batches(self, monkeypatch):
        import json
        from types import SimpleNamespace
        from app.rag import node_3_analyzer as module
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer

        analyzer = object.__new__(Node3RequirementAnalyzer)
        calls = []

        def invoke(messages):
            prompt = messages[-1].content
            batch = json.loads(prompt.split("ITENS: ", 1)[1])
            calls.append(batch)
            return SimpleNamespace(content=json.dumps({"explicacoes": [
                {"id": row["id"], "explicacao":
                 f"Este item trata de {row['item']}. Verifique no edital como se aplica."}
                for row in batch
            ]}))

        monkeypatch.setattr(analyzer, "_invoke_with_rotation", invoke)
        monkeypatch.setattr(module.time, "sleep", lambda _: None)
        agg = {
            "requisitos_participacao": [f"Condição {i}" for i in range(3)],
            "selecao": [f"Critério {i}" for i in range(12)],
        }
        result = analyzer._explain_all_requirements(agg)
        assert [len(batch) for batch in calls] == [3, 5, 5, 2]
        assert [row["item"] for row in result["selecao"]] == agg["selecao"]
        assert [row["id"] for row in result["selecao"]] == list(range(1, 13))

    def test_incomplete_explanation_fails_instead_of_dropping_an_item(self, monkeypatch):
        import pytest
        from types import SimpleNamespace
        from app.rag import node_3_analyzer as module
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer

        analyzer = object.__new__(Node3RequirementAnalyzer)
        monkeypatch.setattr(module.time, "sleep", lambda _: None)
        monkeypatch.setattr(analyzer, "_invoke_with_rotation", lambda messages:
                            SimpleNamespace(content='{"explicacoes":[{"id":1,'
                                                    '"explicacao":"Este item descreve uma condição que deve ser conferida no edital."}]}'))
        agg = {"requisitos_participacao": [], "selecao": ["Primeiro", "Segundo"]}
        with pytest.raises(RuntimeError, match="Explicação incompleta"):
            analyzer._explain_all_requirements(agg)

    def test_retries_only_missing_explanation_ids(self, monkeypatch):
        import json
        from types import SimpleNamespace
        from app.rag import node_3_analyzer as module
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer

        analyzer = object.__new__(Node3RequirementAnalyzer)
        requested = []
        def invoke(messages):
            batch = json.loads(messages[-1].content.split("ITENS: ", 1)[1])
            requested.append([row["id"] for row in batch])
            rows = batch[:1] if len(requested) == 1 else batch
            return SimpleNamespace(content=json.dumps({"explicacoes": [
                {"id": row["id"], "explicacao":
                 "Este item precisa ser conferido no edital para entender sua aplicação."}
                for row in rows
            ]}))

        monkeypatch.setattr(analyzer, "_invoke_with_rotation", invoke)
        monkeypatch.setattr(module.time, "sleep", lambda _: None)
        result = analyzer._explain_all_requirements({
            "requisitos_participacao": ["Um", "Dois", "Três"], "selecao": []
        })
        assert requested == [[1, 2, 3], [2, 3]]
        assert [row["id"] for row in result["participacao"]] == [1, 2, 3]

    def test_resume_after_failed_explanation_reuses_scan_and_completed_ids(
        self, monkeypatch, tmp_path
    ):
        import json
        from types import SimpleNamespace
        from app.rag import node_3_analyzer as module
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer

        monkeypatch.setattr(module.time, "sleep", lambda _: None)
        chunk = {"content": "Atestado exigido na habilitação", "section": "documentacao",
                 "metadata": {"chunk_id": 0, "page": 2}}
        extraction = {
            "objeto": "", "documentos": ["Atestado"], "documentos_execucao": [],
            "anexos_referencia": [], "pendencias_documentais": [],
            "requisitos_participacao": ["Condição A", "Condição B"],
            "prazos": [], "custos": [], "selecao": [], "entregas": [],
            "eliminacao": [], "riscos": [], "nivel_risco": "BAIXO",
        }
        calls = {"scan": 0, "ids": []}
        allow_completion = False

        def invoke(messages, *args):
            prompt = messages[-1].content
            if "ITENS: " not in prompt:
                calls["scan"] += 1
                return SimpleNamespace(content=json.dumps(extraction))
            entries = json.loads(prompt.split("ITENS: ", 1)[1])
            ids = [entry["id"] for entry in entries]
            calls["ids"].append(ids)
            answered = entries if allow_completion else entries[:1] if ids == [1, 2] else []
            return SimpleNamespace(content=json.dumps({"explicacoes": [
                {"id": entry["id"], "explicacao":
                 "Este item descreve uma condição que precisa ser conferida no edital."}
                for entry in answered
            ]}))

        def analyzer():
            node = object.__new__(Node3RequirementAnalyzer)
            node.checkpoint_directory = tmp_path
            node.model_name = "modelo-teste"
            monkeypatch.setattr(node, "_invoke_with_rotation", invoke)
            monkeypatch.setattr(node, "_retrieve_rag_chunks",
                                lambda **kwargs: {"participacao": [chunk]})
            monkeypatch.setattr(node, "_synthesize_explanatory_report",
                                lambda agg, refs: {"resumo": "Resumo", "requisitos": "Requisitos"})
            return node

        with pytest.raises(RuntimeError, match="IDs faltantes"):
            analyzer()._process_with_rag_llm([chunk])
        assert calls["scan"] == 1
        assert calls["ids"] == [[1, 2], [2], [2]]
        assert list(tmp_path.glob("*.json"))

        allow_completion = True
        result = analyzer()._process_with_rag_llm([chunk])
        assert calls["scan"] == 1
        assert calls["ids"][-1] == [2]
        assert len(result["detailed_explanations"]["participacao"]) == 2

    def test_scan_checkpoint_skips_batches_already_completed(self, monkeypatch, tmp_path):
        import json
        from types import SimpleNamespace
        from app.rag import node_3_analyzer as module
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer

        monkeypatch.setattr(module.time, "sleep", lambda _: None)
        chunks = [
            {"content": f"Trecho {i}", "section": "outros", "metadata": {"page": 1}}
            for i in range(6)
        ]
        extraction = {
            "objeto": "", "documentos": [], "documentos_execucao": [],
            "anexos_referencia": [], "pendencias_documentais": [],
            "requisitos_participacao": [], "prazos": [], "custos": [],
            "selecao": [], "entregas": [], "eliminacao": [], "riscos": [],
            "nivel_risco": "BAIXO",
        }
        prompts = []
        def invoke(messages, *args):
            text = messages[-1].content
            prompts.append(text)
            if "Trecho 5" in text and len(prompts) == 2:
                raise RuntimeError("serviço temporariamente indisponível")
            return SimpleNamespace(content=json.dumps(extraction))

        def analyzer():
            node = object.__new__(Node3RequirementAnalyzer)
            node.checkpoint_directory = tmp_path
            node.model_name = "modelo-teste"
            monkeypatch.setattr(node, "_invoke_with_rotation", invoke)
            monkeypatch.setattr(node, "_retrieve_rag_chunks",
                                lambda **kwargs: {"objeto": [chunks[0]]})
            monkeypatch.setattr(node, "_explain_all_requirements",
                                lambda agg, checkpoint, chunks=None: {"participacao": [], "selecao": []})
            return node

        with pytest.raises(RuntimeError, match="lote 2/2"):
            analyzer()._process_with_rag_llm(chunks)
        analyzer()._process_with_rag_llm(chunks)
        assert len(prompts) == 3
        assert "Trecho 0" in prompts[0]
        assert "Trecho 5" in prompts[1] and "Trecho 5" in prompts[2]

    def test_failed_batch_cannot_publish_success(self, monkeypatch):
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer

        analyzer = object.__new__(Node3RequirementAnalyzer)
        def fail(messages, *args):
            raise RuntimeError("limite de API")
        monkeypatch.setattr(analyzer, "_invoke_with_rotation", fail)
        with pytest.raises(RuntimeError, match="Análise incompleta"):
            analyzer._process_with_rag_llm([
                {"content": "Prazo: 30 dias", "metadata": {"page": 1}}
            ])

    def test_real_mode_dispatches_to_unified_llm(self, monkeypatch):
        """O fluxo real usa a passagem única e preserva a saída para o nó 4."""
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer

        analyzer = object.__new__(Node3RequirementAnalyzer)
        analyzer.mock_mode = False
        chunks = [{"content": "Prazo: 30 dias", "section": "prazos"}]
        expected = {
            "total_chunks_processed": 1,
            "structured_analysis": {},
            "critical_analysis": {},
            "llm_analysis": {"prazos": ["30 dias"]},
        }
        monkeypatch.setattr(analyzer, "_process_with_rag_llm", lambda incoming: expected if incoming is chunks else None)

        assert analyzer.process_from_node2(chunks) is expected

    def test_complete_workflow_node2_to_node3(self):
        """Testa workflow completo Nó 2 → Nó 3."""
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer
        
        node3 = Node3RequirementAnalyzer(api_key="test_key", mock_mode=True)
        
        # Chunks do Nó 2 com embeddings
        node2_output = {
            "chunks_with_embeddings": [
                {
                    "content": "CPU i5, 8GB RAM",
                    "section": "exigencias_tecnicas",
                    "embedding": [0.1, 0.2, 0.3]
                },
                {
                    "content": "Prazo 30 dias",
                    "section": "prazos",
                    "embedding": [0.4, 0.5, 0.6]
                }
            ],
            "total_embeddings": 2
        }
        
        result = node3.process_complete_analysis(node2_output)
        
        assert "summary" in result
        assert "technical_analysis" in result
        assert "risk_assessment" in result
        assert "recommendations" in result
    
    def test_combine_structured_and_critical_analysis(self):
        """Testa combinação de análise estruturada e crítica."""
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer
        
        node3 = Node3RequirementAnalyzer(api_key="test_key", mock_mode=True)
        
        chunks = [
            {"content": "Processador i5", "section": "exigencias_tecnicas"},
            {"content": "Prazo curto", "section": "prazos"}
        ]
        
        result = node3.combine_analyses(chunks)
        
        assert "structured_info" in result
        assert "critical_info" in result
        assert "combined_risk_level" in result
        assert "priority_recommendations" in result
    
    def test_generate_executive_summary(self):
        """Testa geração de resumo executivo."""
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer
        
        node3 = Node3RequirementAnalyzer(api_key="test_key", mock_mode=True)
        
        analysis_result = {
            "structured_info": [
                {
                    "technical_requirements": ["CPU i5"],
                    "deadlines": ["30 dias"]
                }
            ],
            "critical_info": [
                {
                    "risk_level": "MEDIO",
                    "critical_points": ["Prazo curto"]
                }
            ],
            "combined_risk_level": "MEDIO",
            "priority_recommendations": ["Verificar prazo"]
        }
        
        summary = node3.generate_executive_summary(analysis_result)
        
        assert "overview" in summary
        assert "key_findings" in summary
        assert "risk_summary" in summary
        assert "next_steps" in summary
    
    def test_validate_integration_output(self):
        """Testa validação da saída da integração."""
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer
        
        node3 = Node3RequirementAnalyzer(api_key="test_key", mock_mode=True)
        
        # Saída válida
        valid_output = {
            "total_chunks_processed": 2,
            "structured_analysis": {"test": "data"},
            "critical_analysis": {"risk_level": "ALTO"}
        }
        assert node3.validate_integration_output(valid_output) is True
        
        # Saída inválida (sem campo obrigatório)
        invalid_output = {
            "structured_analysis": {"test": "data"}
        }
        assert node3.validate_integration_output(invalid_output) is False
    
    def test_empty_chunks_integration(self):
        """Testa integração com chunks vazios."""
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer
        
        node3 = Node3RequirementAnalyzer(api_key="test_key", mock_mode=True)
        
        result = node3.process_from_node2([])
        
        assert result["total_chunks_processed"] == 0
        assert result["structured_analysis"] == {}
        assert result["critical_analysis"] == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
