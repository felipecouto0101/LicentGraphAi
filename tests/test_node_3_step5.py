"""
Testes para Etapa 5 do Nó 3: Integração completa com Nó 2

TDD Approach: Testes escritos antes da implementação
"""

import pytest
from typing import Dict, List


class TestTransientGroqFailures:
    def test_truncated_single_chunk_splits_text_and_resumes_parts(self, monkeypatch, tmp_path):
        import json
        from app.rag import node_3_analyzer as module

        node = object.__new__(module.Node3RequirementAnalyzer)
        node._checkpoint_file = tmp_path / "text-split.json"
        text = ("Primeira cláusula completa sobre documentos obrigatórios. " * 5
                + "Segunda cláusula completa sobre condições de participação. " * 5)
        chunk = {"content": text, "metadata": {"page": 15, "chunk_id": 42}}
        cut = node._extraction_text_cut(text)
        calls, completed = [], []
        failed = [False]
        def extract(batch, system, context):
            fragment = batch[0]
            content = fragment["content"]
            calls.append(content)
            assert fragment["metadata"]["page"] == 15
            assert fragment["metadata"]["chunk_id"] == 42
            if content == text:
                raise module.GroqOutputTruncated("length")
            if content == text[cut:] and not failed[0]:
                failed[0] = True
                assert context[-1]["content"] == text[:cut]
                raise RuntimeError("503")
            completed.append(content)
            quote = content.strip()[:45]
            sources = node._verify_extracted_rules([{
                "tipo": "participacao", "titulo": "Regra do fragmento " + quote,
                "trecho_id": 1, "citacao": quote, "condicao": "",
            }], batch, context)
            return {"documentos": [quote], "objeto": "", "nivel_risco": "BAIXO"}, sources
        monkeypatch.setattr(node, "_extract_verified_batch", extract)
        state = {}
        with pytest.raises(RuntimeError, match="503"):
            node._extract_budgeted_batch([chunk], "system", [], state, 18)
        restored = json.loads(node._checkpoint_file.read_text(encoding="utf-8"))
        result, sources = node._extract_budgeted_batch([chunk], "system", [], restored, 18)
        assert calls == [text, text[:cut], text[cut:], text[cut:]]
        assert "".join(completed) == text  # Nenhum caractere foi descartado.
        assert len(result["documentos"]) == 2
        assert {source["pagina"] for source in sources["participacao"]} == {15}
        assert {source["chunk_id"] for source in sources["participacao"]} == {42}

    def test_recursive_text_subdivision_covers_every_character(self, monkeypatch, tmp_path):
        from app.rag import node_3_analyzer as module

        node = object.__new__(module.Node3RequirementAnalyzer)
        node._checkpoint_file = tmp_path / "recursive.json"
        text = "".join(f"Cláusula {i}: obrigação contratual descrita no edital. " for i in range(20))
        completed = []
        def extract(batch, *args):
            content = batch[0]["content"]
            if len(content) > 150:
                raise module.GroqOutputTruncated("length")
            completed.append(content)
            return {"entregas": [content], "nivel_risco": "BAIXO"}, {"participacao": [], "selecao": []}
        monkeypatch.setattr(node, "_extract_verified_batch", extract)
        state = {}
        result, _ = node._extract_budgeted_batch([{
            "content": text, "metadata": {"page": 7, "chunk_id": 6}
        }], "system", [], state, 18)
        assert "".join(completed) == text
        assert "".join(result["entregas"]) == text
        assert len(completed) > 2
        assert any("split_at" in part for part in state["partial_batches"]["18"].values())

    def test_text_subdivision_stops_for_tiny_truncated_fragment(self, monkeypatch, tmp_path):
        from app.rag import node_3_analyzer as module

        node = object.__new__(module.Node3RequirementAnalyzer)
        node._checkpoint_file = tmp_path / "tiny.json"
        calls = []
        def extract(batch, *args):
            calls.append(batch[0]["content"])
            raise module.GroqOutputTruncated("length")
        monkeypatch.setattr(node, "_extract_verified_batch", extract)
        chunk = {"content": "Uma cláusula curta que sempre retorna resposta truncada.",
                 "metadata": {"page": 3, "chunk_id": 2}}
        with pytest.raises(module.GroqRequestTooLarge, match="subtrecho"):
            node._extract_budgeted_batch([chunk], "system", [], {}, 18)
        assert calls == [chunk["content"]]  # Não repete eternamente.

    def test_quote_across_text_boundary_keeps_original_page_and_chunk(self):
        from app.rag import node_3_analyzer as module

        prefix = "Se a empresa tiver menos de dois anos, "
        suffix = "apresente os balanços disponíveis."
        metadata = {"page": 4, "chunk_id": 9}
        prior = {"content": prefix, "metadata": {**metadata,
                  "source_char_start": 0, "source_char_end": len(prefix)}}
        chunk = {"content": suffix, "metadata": {**metadata,
                  "source_char_start": len(prefix), "source_char_end": len(prefix + suffix)}}
        rows = [{"tipo": "participacao", "titulo": "Balanços de empresas recentes",
                 "citacao": prefix + suffix, "condicao": prefix.strip(), "trecho_id": 1}]
        source = module.Node3RequirementAnalyzer._verify_extracted_rules(rows, [chunk], [prior])
        assert source["participacao"][0]["citacao"] == prefix + suffix
        assert source["participacao"][0]["pagina"] == 4
        assert source["participacao"][0]["chunk_id"] == 9
        prior["metadata"]["page"] = 5
        with pytest.raises(ValueError, match="Citação"):
            module.Node3RequirementAnalyzer._verify_extracted_rules(rows, [chunk], [prior])

    def test_otpm_reduces_output_budget_without_splitting_input(self, monkeypatch):
        from types import SimpleNamespace
        from app.rag import node_3_analyzer as module

        class Limited(Exception):
            status_code = 429
            body = {"error": {"type": "tokens", "message":
                "Request too large on output tokens per minute (OTPM): Limit 1000, Requested 1770"}}
        node = object.__new__(module.Node3RequirementAnalyzer)
        node.model_name = "qwen/qwen3.8-27b"
        node.temperature = 0.3
        node.max_tokens = 2500
        calls = []
        def create(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise Limited("429")
            return SimpleNamespace(headers={}, parse=lambda: SimpleNamespace(choices=[
                SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="ok"))]))
        node.llm = SimpleNamespace(client=SimpleNamespace(
            with_raw_response=SimpleNamespace(create=create)))
        clock = [100.0]
        waits = []
        monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
        def sleep(seconds):
            waits.append(seconds)
            clock[0] += seconds
        monkeypatch.setattr(module.time, "sleep", sleep)
        messages = [module.HumanMessage(content="original input")]
        assert node._invoke_with_rotation(messages).content == "ok"
        assert [call["max_tokens"] for call in calls] == [2500, 950]
        assert calls[1]["reasoning_effort"] == "none"
        assert "reasoning_effort" not in calls[0]
        assert calls[0]["messages"] == calls[1]["messages"]
        assert waits == [61.0]

    def test_truncated_json_is_not_accepted_as_complete(self):
        from types import SimpleNamespace
        from app.rag import node_3_analyzer as module

        node = object.__new__(module.Node3RequirementAnalyzer)
        node.model_name = "model"
        node.temperature = 0.3
        node.max_tokens = 2500
        node._output_token_cap = 950
        calls = []
        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(headers={}, parse=lambda: SimpleNamespace(choices=[
                SimpleNamespace(finish_reason="length", message=SimpleNamespace(content='{"documentos": []}'))]))
        node.llm = SimpleNamespace(client=SimpleNamespace(
            with_raw_response=SimpleNamespace(create=create)))
        with pytest.raises(module.GroqOutputTruncated, match="truncada"):
            node._invoke_groq([module.HumanMessage(content="input")])
        assert calls[0]["max_tokens"] == 950
        assert "reasoning_effort" not in calls[0]

    def test_error_detail_keeps_limit_scope_and_hides_credentials(self):
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer

        class Limited(Exception):
            body = {"error": {"message": "Input tokens per minute: Limit 1000, Requested 1811. "
                                         "Key gsk_exampleSecret Bearer otherSecret"}}
        detail = Node3RequirementAnalyzer._groq_error_detail(Limited())
        assert "Input tokens per minute: Limit 1000, Requested 1811" in detail
        assert "exampleSecret" not in detail
        assert "otherSecret" not in detail

    def test_oversized_token_request_is_not_retried_unchanged(self, monkeypatch):
        from types import SimpleNamespace
        from app.rag import node_3_analyzer as module

        class TooLarge(Exception):
            status_code = 429
            body = {"error": {"type": "tokens", "code": "rate_limit_exceeded",
                              "message": "Tokens per minute: Limit 8000, Used 0, Requested 9120"}}
            response = SimpleNamespace(headers={"x-ratelimit-limit-tokens": "8000",
                                                "x-ratelimit-remaining-tokens": "8000"})
        node = object.__new__(module.Node3RequirementAnalyzer)
        node.llm = SimpleNamespace(invoke=lambda _: (_ for _ in ()).throw(TooLarge("429")))
        monkeypatch.setattr(module.time, "sleep", lambda _: pytest.fail("should resize, not wait"))
        with pytest.raises(module.GroqRequestTooLarge):
            node._invoke_with_rotation(["prompt"])

    def test_split_batch_resumes_saved_parts_without_losing_chunks(self, monkeypatch, tmp_path):
        import json
        from app.rag import node_3_analyzer as module

        node = object.__new__(module.Node3RequirementAnalyzer)
        node._checkpoint_file = tmp_path / "split.json"
        chunks = [{"content": f"Trecho {i}", "metadata": {"page": 3, "chunk_id": i}}
                  for i in range(4)]
        calls = []
        def extract(batch, *args):
            ids = [chunk["metadata"]["chunk_id"] for chunk in batch]
            calls.append(ids)
            if len(batch) > 2:
                raise module.GroqRequestTooLarge("large")
            if ids == [2, 3] and calls.count(ids) == 1:
                raise RuntimeError("503")
            return ({"documentos": [f"D{i}" for i in ids], "objeto": "Objeto",
                     "nivel_risco": "BAIXO"}, {"participacao": [], "selecao": []})
        monkeypatch.setattr(node, "_extract_verified_batch", extract)
        state = {}
        with pytest.raises(RuntimeError, match="503"):
            node._extract_budgeted_batch(chunks, "system", [], state, 6)
        restored = json.loads(node._checkpoint_file.read_text(encoding="utf-8"))
        result, _ = node._extract_budgeted_batch(chunks, "system", [], restored, 6)
        assert calls == [[0, 1, 2, 3], [0, 1], [2, 3], [2, 3]]
        assert result["documentos"] == ["D0", "D1", "D2", "D3"]

    def test_raw_response_tracks_remaining_quota_without_extra_call(self):
        from types import SimpleNamespace
        from app.rag import node_3_analyzer as module

        analyzer = object.__new__(module.Node3RequirementAnalyzer)
        analyzer.model_name = "qwen/qwen3.8-27b"
        analyzer.temperature = 0.3
        analyzer.max_tokens = 2500
        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                headers={"x-ratelimit-remaining-tokens": "3500",
                         "x-ratelimit-limit-tokens": "8000",
                         "x-ratelimit-reset-tokens": "31s",
                         "x-ratelimit-remaining-requests": "44"},
                parse=lambda: SimpleNamespace(choices=[SimpleNamespace(
                    message=SimpleNamespace(content='{"ok": true}'))]),
            )

        analyzer.llm = SimpleNamespace(client=SimpleNamespace(
            with_raw_response=SimpleNamespace(create=create)))
        response = analyzer._invoke_with_rotation([module.HumanMessage(content="teste")])
        assert response.content == '{"ok": true}'
        assert len(calls) == 1
        assert analyzer._quota["tokens_remaining"] == 3500
        assert analyzer._quota["requests_remaining"] == 44

    def test_waits_for_token_reset_before_next_request(self, monkeypatch):
        from types import SimpleNamespace
        from app.rag import node_3_analyzer as module

        analyzer = object.__new__(module.Node3RequirementAnalyzer)
        analyzer.max_tokens = 2500
        clock = [100.0]
        waits = []
        analyzer._quota = {"tokens_remaining": 1200, "tokens_limit": 8000,
                           "tokens_reset_at": 130.0, "requests_remaining": 20}
        analyzer.llm = SimpleNamespace(invoke=lambda _: SimpleNamespace(content="ok"))
        monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
        def sleep(seconds):
            waits.append(seconds)
            clock[0] += seconds
        monkeypatch.setattr(module.time, "sleep", sleep)
        assert analyzer._invoke_with_rotation(["prompt"]).content == "ok"
        assert waits == [31.0]
        assert analyzer._quota["tokens_remaining"] == 8000

    def test_daily_request_limit_stops_before_sending(self, monkeypatch):
        from types import SimpleNamespace
        from app.rag import node_3_analyzer as module

        analyzer = object.__new__(module.Node3RequirementAnalyzer)
        analyzer._quota = {"requests_remaining": 0, "requests_reset_at": 5000.0}
        analyzer.llm = SimpleNamespace(invoke=lambda _: pytest.fail("unexpected API call"))
        monkeypatch.setattr(module.time, "monotonic", lambda: 100.0)
        with pytest.raises(RuntimeError, match="requisições por dia esgotada"):
            analyzer._invoke_with_rotation(["prompt"])

    def test_local_rpm_budget_paces_calls(self, monkeypatch):
        from types import SimpleNamespace
        from app.rag import node_3_analyzer as module

        analyzer = object.__new__(module.Node3RequirementAnalyzer)
        clock = [100.0]
        waits = []
        analyzer.llm = SimpleNamespace(invoke=lambda _: SimpleNamespace(content="ok"))
        monkeypatch.setenv("GROQ_RPM_BUDGET", "2")
        monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
        def sleep(seconds):
            waits.append(seconds)
            clock[0] += seconds
        monkeypatch.setattr(module.time, "sleep", sleep)
        for _ in range(3):
            assert analyzer._invoke_with_rotation(["prompt"]).content == "ok"
        assert waits == [61.0]

    def test_429_reduces_local_rpm_for_later_calls(self, monkeypatch):
        from types import SimpleNamespace
        from app.rag import node_3_analyzer as module

        class RateLimit(Exception):
            status_code = 429
            response = SimpleNamespace(headers={"retry-after": "1"})

        clock = [100.0]
        waits = []
        calls = []
        analyzer = object.__new__(module.Node3RequirementAnalyzer)
        def invoke(_):
            calls.append(clock[0])
            if len(calls) == 1:
                raise RateLimit("429")
            return SimpleNamespace(content="ok")
        analyzer.llm = SimpleNamespace(invoke=invoke)
        monkeypatch.setenv("GROQ_RPM_BUDGET", "10")
        monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
        def sleep(seconds):
            waits.append(seconds)
            clock[0] += seconds
        monkeypatch.setattr(module.time, "sleep", sleep)

        analyzer._invoke_with_rotation(["prompt"])
        analyzer._invoke_with_rotation(["next prompt"])
        assert analyzer._rpm_budget_override == 1
        assert waits == [2.0, 59.0, 2.0]

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
        with pytest.raises(RuntimeError, match="limite de uso"):
            analyzer._invoke_with_rotation(["message"])

        assert len(calls) == 4
        assert waits == [60, 60, 60]
        assert analyzer._current_key_idx == 0

    def test_429_obeys_retry_after_before_retrying(self, monkeypatch):
        from types import SimpleNamespace
        from app.rag import node_3_analyzer as module

        class RateLimit(Exception):
            status_code = 429
            response = SimpleNamespace(headers={"retry-after": "95"})

        analyzer = object.__new__(module.Node3RequirementAnalyzer)
        analyzer.llm = SimpleNamespace(invoke=lambda _: (_ for _ in ()).throw(RateLimit("429")))
        waits = []
        monkeypatch.setattr(module.time, "sleep", waits.append)

        with pytest.raises(RuntimeError, match="limite de uso"):
            analyzer._invoke_with_rotation(["message"])
        assert waits == [96, 96, 96]

    def test_long_retry_after_reports_wait_without_sending_again(self, monkeypatch):
        from types import SimpleNamespace
        from app.rag import node_3_analyzer as module

        class RateLimit(Exception):
            status_code = 429
            response = SimpleNamespace(headers={"retry-after": "7200"})

        calls = []
        analyzer = object.__new__(module.Node3RequirementAnalyzer)
        def invoke(_):
            calls.append(1)
            raise RateLimit("429")
        analyzer.llm = SimpleNamespace(invoke=invoke)
        monkeypatch.setattr(module.time, "sleep", lambda _: pytest.fail("unexpected wait"))

        with pytest.raises(RuntimeError, match="7201s"):
            analyzer._invoke_with_rotation(["message"])
        assert len(calls) == 1

    def test_retry_delay_from_message_keeps_full_duration(self):
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer

        assert Node3RequirementAnalyzer._rate_limit_wait(
            Exception("Please try again in 2m30.5s")) == 151.5

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
    def test_rules_require_literal_quote_from_declared_page(self):
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer

        batch = [{"content": "O custo do licitante ultrapassa o valor da proposta.",
                  "metadata": {"page": 15, "chunk_id": 7}},
                 {"content": "Caso seja adotado outro procedimento, muda a etapa.",
                  "metadata": {"page": 16, "chunk_id": 8}}]
        row = {"tipo": "selecao", "titulo": "Verificação da exequibilidade",
               "trecho_id": 1, "citacao": batch[0]["content"], "condicao": ""}
        result = Node3RequirementAnalyzer._verify_extracted_rules([row], batch)
        assert result["selecao"][0]["pagina"] == 15
        assert result["selecao"][0]["citacao"] == batch[0]["content"]
        with pytest.raises(ValueError, match="Citação"):
            Node3RequirementAnalyzer._verify_extracted_rules([
                {**row, "citacao": "O preço da proposta ultrapassa o valor da proposta."}
            ], batch)
        with pytest.raises(ValueError, match="Condição"):
            Node3RequirementAnalyzer._verify_extracted_rules([
                {**row, "condicao": batch[1]["content"]}
            ], batch)

    def test_relocates_literal_quote_to_other_chunk_on_same_page(self):
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer

        batch = [
            {"content": "Critério de menor preço global.",
             "metadata": {"page": 8, "chunk_id": 11}},
            {"content": "Será permitida nova proposta após a disputa.",
             "metadata": {"page": 8, "chunk_id": 12}},
        ]
        row = {"tipo": "selecao", "titulo": "Nova proposta após disputa",
               "trecho_id": 1, "citacao": batch[1]["content"], "condicao": ""}
        result = Node3RequirementAnalyzer._verify_extracted_rules([row], batch)
        assert result["selecao"][0]["chunk_id"] == 12
        assert result["selecao"][0]["pagina"] == 8
        batch[1]["metadata"]["page"] = 9
        with pytest.raises(ValueError, match="Citação"):
            Node3RequirementAnalyzer._verify_extracted_rules([row], batch)

    def test_invalid_source_is_retried_before_saving_batch(self, monkeypatch):
        from types import SimpleNamespace
        from app.rag import node_3_analyzer as module

        node = object.__new__(module.Node3RequirementAnalyzer)
        chunk = {"content": "O fornecedor deve manter seu cadastro atualizado.",
                 "metadata": {"page": 3}}
        calls = []

        def invoke(messages):
            import json
            calls.append(messages)
            quote = ("Cláusula inventada sem suporte no PDF."
                     if len(calls) == 1 else chunk["content"])
            return SimpleNamespace(content=json.dumps({
                "documentos": [], "documentos_execucao": [], "anexos_referencia": [],
                "pendencias_documentais": [], "prazos": [], "custos": [],
                "entregas": [], "eliminacao": [], "riscos": [],
                "regras": [{"tipo": "participacao", "titulo": "Cadastro atualizado",
                            "trecho_id": 1, "citacao": quote, "condicao": ""}],
            }))

        monkeypatch.setattr(node, "_invoke_with_rotation", invoke)
        monkeypatch.setattr(module.time, "sleep", lambda _: None)
        _, sources = node._extract_verified_batch([chunk], "system")
        assert len(calls) == 2
        assert sources["participacao"][0]["pagina"] == 3

    def test_persistent_invalid_quote_is_flagged_without_losing_verified_rule(self, monkeypatch):
        import json
        from types import SimpleNamespace
        from app.rag import node_3_analyzer as module

        node = object.__new__(module.Node3RequirementAnalyzer)
        chunk = {"content": "O cadastro deverá permanecer atualizado durante a disputa.",
                 "metadata": {"page": 3, "chunk_id": 9}}
        calls = []
        def invoke(_):
            calls.append(1)
            return SimpleNamespace(content=json.dumps({
                "documentos": [], "documentos_execucao": [], "anexos_referencia": [],
                "pendencias_documentais": [], "prazos": [], "custos": [],
                "entregas": [], "eliminacao": [], "riscos": [],
                "regras": [
                    {"tipo": "participacao", "titulo": "Cadastro atualizado durante a disputa",
                     "trecho_id": 1, "citacao": chunk["content"], "condicao": ""},
                    {"tipo": "selecao", "titulo": "Outra etapa inventada",
                     "trecho_id": 1, "citacao": "Será exigida uma etapa inexistente.",
                     "condicao": ""},
                ],
            }))
        monkeypatch.setattr(node, "_invoke_with_rotation", invoke)
        monkeypatch.setattr(module.time, "sleep", lambda _: None)
        parsed, sources = node._extract_verified_batch([chunk], "system")
        assert len(calls) == 3
        assert len(sources["participacao"]) == 1
        assert sources["selecao"] == []
        assert parsed["_unverified_rules"][0]["titulo_proposto"] == "Outra etapa inventada"
        assert parsed["_unverified_rules"][0]["pagina_sugerida"] == 3

    def test_pending_rule_is_saved_and_visible_after_resume(self, monkeypatch, tmp_path):
        from app.rag import node_3_analyzer as module

        chunk = {"content": "O cadastro deverá permanecer atualizado durante a disputa.",
                 "metadata": {"page": 3, "chunk_id": 9}}
        node = object.__new__(module.Node3RequirementAnalyzer)
        node.model_name = "qwen/qwen3.8-27b"
        node.checkpoint_directory = tmp_path
        calls = []
        def extract(*args):
            calls.append(1)
            return ({
                "objeto": "", "documentos": [], "documentos_execucao": [],
                "anexos_referencia": [], "pendencias_documentais": [],
                "prazos": [], "custos": [], "entregas": [], "eliminacao": [],
                "riscos": [], "nivel_risco": "BAIXO",
                "_unverified_rules": [{"titulo_proposto": "Etapa incerta",
                                       "pagina_sugerida": 3, "motivo": "Citação ausente"}],
            }, {"participacao": [], "selecao": []})
        monkeypatch.setattr(node, "_extract_verified_batch", extract)
        monkeypatch.setattr(node, "_retrieve_rag_chunks", lambda **kwargs: {"prazos": [chunk]})
        monkeypatch.setattr(node, "_explain_all_requirements", lambda *args: {
            "participacao": [], "selecao": []})
        first = node._process_with_rag_llm([chunk])
        assert first["unverified_rules"][0]["lote"] == 1
        assert first["selection_process"] == []
        assert node._checkpoint_file.exists()
        second = node._process_with_rag_llm([chunk])
        assert len(calls) == 1
        assert second["unverified_rules"] == first["unverified_rules"]

    def test_invalid_condition_gets_specific_feedback_on_retry(self, monkeypatch):
        import json
        from types import SimpleNamespace
        from app.rag import node_3_analyzer as module

        node = object.__new__(module.Node3RequirementAnalyzer)
        chunk = {"content": "Neste procedimento, será permitida nova proposta.",
                 "metadata": {"page": 3}}
        parent = {"content": "Caso seja adotada outra modalidade.",
                  "metadata": {"page": 3}}
        requests = []

        def invoke(messages):
            requests.append(messages[-1].content)
            return SimpleNamespace(content=json.dumps({
                "documentos": [], "documentos_execucao": [], "anexos_referencia": [],
                "pendencias_documentais": [], "prazos": [], "custos": [],
                "entregas": [], "eliminacao": [], "riscos": [],
                "regras": [{"tipo": "selecao", "titulo": "Nova proposta na outra modalidade",
                            "trecho_id": 1, "citacao": chunk["content"],
                            "condicao": ("Modalidade inventada" if len(requests) == 1
                                         else "Caso seja adotada outra modalidade") }],
            }))

        monkeypatch.setattr(node, "_invoke_with_rotation", invoke)
        monkeypatch.setattr(module.time, "sleep", lambda _: None)
        _, sources = node._extract_verified_batch([chunk], "system", [parent])
        assert len(requests) == 2
        assert "Condição da regra não aparece" in requests[1]
        assert sources["selecao"][0]["condicao"] == parent["content"].rstrip(".")

    def test_literal_conditional_quote_repairs_paraphrased_condition(self):
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer

        quote = "Caso seja adotada outra modalidade, será permitida nova proposta."
        row = {"tipo": "selecao", "titulo": "Nova proposta em modalidade alternativa",
               "trecho_id": 1, "citacao": quote,
               "condicao": "Se escolherem uma modalidade diferente"}
        result = Node3RequirementAnalyzer._verify_extracted_rules(
            [row], [{"content": quote, "metadata": {"page": 3}}]
        )
        assert result["selecao"][0]["condicao"] == quote

    def test_condition_can_come_from_previous_chunk_on_same_page(self):
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer

        parent = {"content": "Caso seja adotado o modo de disputa “aberto e fechado”",
                  "metadata": {"page": 11}}
        child = {"content": "Neste procedimento, poderá manter o lance ou ofertar outro.",
                 "metadata": {"page": 11}}
        row = {"tipo": "selecao", "titulo": "Opção de lance na etapa alternativa",
               "trecho_id": 1, "citacao": child["content"],
               "condicao": parent["content"]}
        result = Node3RequirementAnalyzer._verify_extracted_rules(
            [row], [child], [parent]
        )
        assert result["selecao"][0]["condicao"] == parent["content"]
        parent["metadata"]["page"] = 10
        with pytest.raises(ValueError, match="Condição"):
            Node3RequirementAnalyzer._verify_extracted_rules([row], [child], [parent])

    def test_mode_comparison_uses_this_pdf_header_and_preserves_alternative(self):
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer

        choose = Node3RequirementAnalyzer._mode_applicability
        assert choose('Caso seja adotado o modo de disputa “fechado e aberto”',
                      'MODO DE DISPUTA: ABERTO\n') == "nao_aplicavel"
        assert choose('Caso seja adotado o modo de disputa “aberto”',
                      'MODO DE DISPUTA: ABERTO\n') == "aplicavel"
        assert choose('Se houver lances adicionais', 'Documento sem modalidade') is None

    def test_explanation_uses_extracted_clause_and_marks_alternative(self, monkeypatch):
        import json
        from types import SimpleNamespace
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer

        node = object.__new__(Node3RequirementAnalyzer)
        condition = 'Caso seja adotado o modo de disputa “fechado e aberto”'
        quote = condition + ', poderão ser apresentados lances sucessivos.'
        chunks = [
            {"content": "MODO DE DISPUTA: ABERTO\n", "metadata": {"page": 1, "chunk_id": 0}},
            {"content": quote, "metadata": {"page": 2, "chunk_id": 1}},
        ]
        agg = {"requisitos_participacao": [], "selecao": ["Lances no modo fechado e aberto"]}
        source = {"item": agg["selecao"][0], "pagina": 2, "citacao": quote,
                  "condicao": condition, "chunk_id": 1}

        def invoke(messages):
            batch = json.loads(messages[-1].content.split("ITENS: ", 1)[1])
            assert batch[0]["fonte"]["pagina"] == 2
            return SimpleNamespace(content=json.dumps({"explicacoes": [{
                "id": 1, "situacao": "aplicavel", "evidencia": {"pagina": 99, "trecho": "falso"},
                "explicacao": "Nesse modo de disputa os participantes podem oferecer lances sucessivos na etapa prevista.",
            }]}))

        monkeypatch.setattr(node, "_invoke_with_rotation", invoke)
        result = node._explain_all_requirements(
            agg, chunks=chunks, rule_sources={"participacao": [], "selecao": [source]}
        )["selecao"][0]
        assert result["situacao"] == "nao_aplicavel"
        assert result["evidencia"] == {"pagina": 2, "trecho": quote}
        assert result["condicao"] == condition

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

    def test_old_checkpoint_is_preserved_but_reextracts_missing_sources(self, tmp_path):
        import hashlib
        import json
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer

        node = object.__new__(Node3RequirementAnalyzer)
        node.checkpoint_directory = tmp_path
        node.model_name = "modelo-teste"
        chunks = [{"content": "Dados cadastrais atualizados", "section": "cadastro",
                   "metadata": {"page": 1}}]
        agg = {"requisitos_participacao": []}
        payload = json.dumps({"version": 1, "model": "modelo-teste",
                              "chunks": [(chunks[0]["content"], chunks[0]["section"], 1)]},
                             ensure_ascii=False, sort_keys=True)
        old_sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        old_path = tmp_path / f"{old_sha}.json"
        old_path.write_text(json.dumps({"version": 1, "next_batch": 1,
                                        "agg": {"requisitos_participacao": ["Dados cadastrais atualizados"]}}),
                            encoding="utf-8")

        resumed = node._load_checkpoint(chunks, 2, agg)

        assert resumed["next_batch"] == 0
        assert resumed["agg"]["requisitos_participacao"] == []
        assert resumed["explanations"] == {"participacao": {}, "selecao": {}}
        assert old_path.exists()
        assert node._checkpoint_file != old_path


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
        chunk = {"content": "Registro técnico e menor preço global. Início " + ("x" * 700) + " SENTINELA_FIM",
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
            "regras": [
                {"tipo": "participacao", "titulo": "Registro técnico", "trecho_id": 1,
                 "citacao": "Registro técnico e menor preço global.", "condicao": ""},
                {"tipo": "selecao", "titulo": "Menor preço global", "trecho_id": 1,
                 "citacao": "Registro técnico e menor preço global.", "condicao": ""},
                {"tipo": "selecao", "titulo": "menor preco global!", "trecho_id": 1,
                 "citacao": "Registro técnico e menor preço global.", "condicao": ""},
            ],
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
                            lambda agg, checkpoint=None, chunks=None, rule_sources=None: {"participacao": [], "selecao": []})

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
            "regras": [
                {"tipo": "participacao", "titulo": label, "trecho_id": 1,
                 "citacao": "Atestado exigido na habilitação", "condicao": ""}
                for label in ("Condição A", "Condição B")
            ],
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
            "regras": [],
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
                                lambda agg, checkpoint, chunks=None, rule_sources=None: {"participacao": [], "selecao": []})
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
