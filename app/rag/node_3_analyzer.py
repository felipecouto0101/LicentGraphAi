import logging
import os
import re
import time
import unicodedata
import hashlib
import json
import tempfile
from collections import deque
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Callback global de progresso — preenchido pela API quando há um job ativo
_progress_callback = None


class GroqRequestTooLarge(RuntimeError):
    """O pedido precisa ser reduzido; esperar não altera seu tamanho."""


class GroqOutputTruncated(GroqRequestTooLarge):
    """Resposta interrompida pelo orçamento de saída; não é extração completa."""


class Node3RequirementAnalyzer:
    """
    Nó 3: Análise de Requisitos com IA (Groq + Llama 3.1)

    Responsável por:
    - Configurar conexão com Groq API
    - Inicializar modelo Llama 3.1
    - Analisar requisitos de editais
    """

    # O esquema 2 exige origem literal para cada regra. Checkpoints antigos
    # continuam no disco, mas não podem comprovar a origem de seus títulos.
    CHECKPOINT_VERSION = 2
    # Mudanças no texto das explicações não devem descartar os lotes de extração.
    EXPLANATION_VERSION = 5
    OLD_PENDING_EXPLANATION = (
        "Item extraído sem interpretação confirmada pelos trechos selecionados; "
        "requer revisão da cláusula original."
    )

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "qwen/qwen3.8-27b",
        temperature: float = 0.3,
        max_tokens: int = 2500,
        mock_mode: bool = False,
        persist_directory: str = "./data/vector_db",
        collection_name: str = "licitacoes",
    ):
        """
        Inicializa o Nó 3.

        Args:
            api_key: Chave da API Groq (ou usa variável de ambiente)
            model_name: Nome do modelo LLM
            temperature: Temperatura para geração
            max_tokens: Máximo de tokens na resposta
            mock_mode: Se True, não inicializa LLM real (para testes)
            persist_directory: Diretório do ChromaDB (para busca RAG)
            collection_name: Nome da coleção ChromaDB
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.mock_mode = mock_mode
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.checkpoint_directory = (
            Path(os.getenv("LICIT_CHECKPOINT_DIR", "./data/checkpoints"))
            if not mock_mode else None
        )

        if not self.api_key and not mock_mode:
            raise ValueError(
                "API key é obrigatória. Forneça api_key ou configure GROQ_API_KEY"
            )

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._output_token_cap = max(1, int(os.getenv("GROQ_OUTPUT_TOKEN_BUDGET", "950")))

        if self._output_budget() <= 1000:
            self._rpm_budget_override = 1

        if not mock_mode:
            self._api_keys = self._load_api_keys()
            self._current_key_idx = 0
            self.llm = self._make_llm(self._api_keys[0])
            logger.info(f"Nó 3: {len(self._api_keys)} chave(s) API carregada(s)")
        else:
            self._api_keys = []
            self.llm = None
            logger.info("Nó 3 em modo mock (sem LLM real)")

        logger.info(
            f"Nó 3 inicializado: modelo={model_name}, temperature={temperature}, mock_mode={mock_mode}"
        )

    def _load_api_keys(self) -> list[str]:
        """Carrega todas as chaves API disponíveis do ambiente."""
        keys = []
        # GROQ_API_KEY, GROQ_API_KEY_2, GROQ_API_KEY_3, ...
        for suffix in ["", "_2", "_3", "_4", "_5"]:
            k = os.getenv(f"GROQ_API_KEY{suffix}", "").strip()
            if k and k != "your_groq_api_key_here" and k not in keys:
                keys.append(k)
        if not keys and self.api_key:
            keys.append(self.api_key)
        return keys

    def _output_budget(self) -> int:
        return min(getattr(self, "max_tokens", 2500),
                   getattr(self, "_output_token_cap", 2500))

    def _make_llm(self, api_key: str) -> ChatGroq:
        """Cria uma instância do LLM com a chave fornecida."""
        return ChatGroq(
            model_name=self.model_name,
            api_key=api_key,
            temperature=self.temperature,
            max_tokens=self._output_budget(),
            max_retries=0,  # O nó controla esperas e tentativas usando os cabeçalhos.
        )

    def _rotate_key(self) -> bool:
        """
        Tenta rotacionar para a próxima chave API disponível.
        Retorna True se conseguiu, False se não há mais chaves.
        """
        next_idx = self._current_key_idx + 1
        if next_idx >= len(self._api_keys):
            logger.error("Todas as chaves API esgotaram o limite. Sem mais chaves disponíveis.")
            return False
        self._current_key_idx = next_idx
        self.llm = self._make_llm(self._api_keys[next_idx])
        logger.warning(f"Rotacionando para chave {next_idx + 1}/{len(self._api_keys)}")
        return True

    def _independent_accounts(self) -> bool:
        # Ative apenas para chaves de organizações diferentes: a Groq compartilha
        # os limites entre todas as chaves de uma mesma organização.
        return (os.getenv("GROQ_INDEPENDENT_ACCOUNTS", "false").lower() == "true"
                and len(getattr(self, "_api_keys", [])) > 1)

    def _save_account(self):
        states = getattr(self, "_account_states", None)
        if states is None:
            states = self._account_states = {}
        state = states.setdefault(self._current_key_idx, {})
        state.update(quota=getattr(self, "_quota", {}),
                     calls=getattr(self, "_request_times", deque()),
                     output_cap=getattr(self, "_output_token_cap", 950),
                     rpm=getattr(self, "_rpm_budget_override", 10))
        return state

    def _activate_account(self, index):
        if index == self._current_key_idx:
            return
        self._save_account()
        state = self._account_states.setdefault(index, {
            "quota": {}, "calls": deque(),
            "output_cap": max(1, int(os.getenv("GROQ_OUTPUT_TOKEN_BUDGET", "950"))),
            "rpm": 1 if self._output_budget() <= 1000 else 10,
        })
        self._current_key_idx = index
        self._quota, self._request_times = state["quota"], state["calls"]
        self._output_token_cap, self._rpm_budget_override = state["output_cap"], state["rpm"]
        self.llm = self._make_llm(self._api_keys[index])
        logger.info("Groq: usando conta %s/%s", index + 1, len(self._api_keys))

    def _select_available_account(self, messages):
        """Usa outra organização antes de esperar; preserva a atual se disponível."""
        self._save_account()
        while True:
            now = time.monotonic()
            candidates = []
            oversized = 0
            for offset in range(len(self._api_keys)):
                index = (self._current_key_idx + offset) % len(self._api_keys)
                state = self._account_states.setdefault(index, {
                    "quota": {}, "calls": deque(),
                    "output_cap": max(1, int(os.getenv("GROQ_OUTPUT_TOKEN_BUDGET", "950"))),
                    "rpm": 1 if self._output_budget() <= 1000 else 10,
                })
                quota, calls = state["quota"], state["calls"]
                for kind in ("tokens", "requests"):
                    if quota.get(kind + "_reset_at", float("inf")) <= now:
                        quota.pop(kind + "_remaining", None)
                        quota.pop(kind + "_reset_at", None)
                estimated = sum((len(str(getattr(m, "content", m))) + 2) // 3 + 8
                                for m in messages) + min(self.max_tokens, state["output_cap"])
                if quota.get("tokens_limit") and estimated > quota["tokens_limit"]:
                    oversized += 1
                    continue
                ready = state.get("cooldown_until", 0)
                if quota.get("requests_remaining") == 0:
                    ready = max(ready, quota.get("requests_reset_at", float("inf")))
                if quota.get("tokens_remaining", estimated) < estimated:
                    ready = max(ready, quota.setdefault("tokens_reset_at", now + 60) + 1)
                while calls and now - calls[0] >= 60:
                    calls.popleft()
                rpm = min(max(1, int(os.getenv("GROQ_RPM_BUDGET", "10"))), state["rpm"])
                if len(calls) >= rpm:
                    ready = max(ready, calls[0] + 61)
                if ready <= now:
                    self._activate_account(index)
                    return
                candidates.append(ready)
            if oversized == len(self._api_keys):
                raise GroqRequestTooLarge("Pedido excede a cota de tokens das contas disponíveis.")
            wait = min(candidates, default=float("inf")) - now
            if wait > 300:
                raise RuntimeError("Groq: nenhuma conta disponível neste momento; "
                                   "retome depois. O checkpoint continua salvo.")
            logger.info("Groq: todas as contas em espera; próxima janela em %.0fs", wait)
            time.sleep(min(60, max(1, wait)))

    @staticmethod
    def _rate_limit_wait(error: Exception) -> float:
        """Usa o prazo comunicado pela Groq antes de recorrer a uma espera padrão."""
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None) or {}
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after is not None:
            try:
                return max(1.0, float(retry_after) + 1.0)
            except (TypeError, ValueError):
                pass

        match = re.search(
            r"try again in\s+(?:(\d+)m)?([\d.]+)s",
            str(error), re.IGNORECASE,
        )
        if match:
            try:
                return max(1.0, int(match.group(1) or 0) * 60 + float(match.group(2)) + 1)
            except ValueError:
                pass
        return 60.0

    @staticmethod
    def _groq_error_detail(error: Exception) -> str:
        """Mostra a explicação da API, sem imprimir chave ou corpo do pedido."""
        body = getattr(error, "body", None)
        details = body.get("error", body) if isinstance(body, dict) else {}
        detail = details.get("message") if isinstance(details, dict) else None
        if not isinstance(detail, str) or not detail.strip():
            return "A exceção não forneceu o campo message da Groq."
        detail = re.sub(r"\bgsk_[A-Za-z0-9_-]+", "[chave ocultada]", detail)
        detail = re.sub(r"(?i)\bBearer\s+\S+", "Bearer [ocultado]", detail)
        return " ".join(detail.split())[:1500]

    @staticmethod
    def _reset_seconds(value: str | None) -> float | None:
        if not value:
            return None
        match = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?([\d.]+)s", value.strip())
        if match:
            return (int(match.group(1) or 0) * 3600
                    + int(match.group(2) or 0) * 60 + float(match.group(3)))
        return None

    def _record_quota(self, headers) -> None:
        """Guarda cotas reais retornadas pela API, sem registrar credenciais."""
        if not headers:
            return
        quota = getattr(self, "_quota", None)
        if quota is None:
            quota = self._quota = {}
        now = time.monotonic()
        for kind in ("tokens", "requests"):
            remaining = headers.get(f"x-ratelimit-remaining-{kind}")
            limit = headers.get(f"x-ratelimit-limit-{kind}")
            reset = self._reset_seconds(headers.get(f"x-ratelimit-reset-{kind}"))
            for name, value in (("remaining", remaining), ("limit", limit)):
                try:
                    if value is not None:
                        quota[f"{kind}_{name}"] = max(0, int(value))
                except (TypeError, ValueError):
                    pass
            if reset is not None:
                quota[f"{kind}_reset_at"] = now + reset
        if "tokens_remaining" in quota or "requests_remaining" in quota:
            # Numeric quota counters and account index only; no API credentials.
            # nosemgrep: python.lang.security.audit.logging.logger-credential-leak.python-logger-credential-disclosure
            logger.info("Cota Groq (chave %s): tokens/min restantes=%s; requisições/dia restantes=%s",
                        getattr(self, "_current_key_idx", 0) + 1,
                        quota.get("tokens_remaining", "?"),
                        quota.get("requests_remaining", "?"))

    def _before_groq_request(self, messages: list, enforce_rpm: bool = True) -> None:
        """Espera pela janela de tokens e limita as chamadas por minuto."""
        quota = getattr(self, "_quota", None)
        if quota is None:
            quota = self._quota = {}
        calls = getattr(self, "_request_times", None)
        if calls is None:
            calls = self._request_times = deque()

        # O header de requests representa RPD, não RPM. Um teto local
        # conservador cobre RPM até o primeiro 429, sem supor a cota da conta.
        rpm = max(1, int(os.getenv("GROQ_RPM_BUDGET", "10")))
        rpm = min(rpm, getattr(self, "_rpm_budget_override", rpm))
        now = time.monotonic()
        while calls and now - calls[0] >= 60:
            calls.popleft()
        while enforce_rpm and len(calls) >= rpm:
            wait = max(0.0, 61 - (now - calls[0]))
            logger.info("Limite local de requisições: aguardando %.0fs", wait)
            time.sleep(wait)
            now = time.monotonic()
            while calls and now - calls[0] >= 60:
                calls.popleft()

        if quota.get("requests_remaining") == 0:
            reset_at = quota.get("requests_reset_at")
            if reset_at and now >= reset_at:
                quota.pop("requests_remaining", None)
            else:
                wait = max(0.0, reset_at - now) if reset_at else None
                raise RuntimeError(
                    "Groq: cota de requisições por dia esgotada"
                    + (f"; tente novamente em {wait:.0f}s" if wait else "")
                    + ". O checkpoint continua salvo."
                )

        # Estimativa conservadora: não é a tokenização exata do modelo.
        # Reserva também espaço para a resposta configurada.
        estimated = sum((len(str(getattr(m, "content", m))) + 2) // 3 + 8
                        for m in messages)
        estimated += self._output_budget()
        limit = quota.get("tokens_limit")
        if limit and estimated > limit:
            raise GroqRequestTooLarge(
                f"A chamada pode exigir cerca de {estimated} tokens, acima da cota "
                f"de {limit} tokens/min informada pela Groq. Reduza o lote ou use "
                "um limite maior; o checkpoint continua salvo."
            )
        remaining = quota.get("tokens_remaining")
        reset_at = quota.get("tokens_reset_at")
        if remaining is not None and remaining < estimated and reset_at:
            wait = max(0.0, reset_at - now) + 1
            if wait > 300:
                raise RuntimeError(
                    f"Groq: aguarde {wait:.0f}s pela renovação da cota de tokens; "
                    "o checkpoint continua salvo."
                )
            # Token counts and wait duration only; no API credentials.
            # nosemgrep: python.lang.security.audit.logging.logger-credential-leak.python-logger-credential-disclosure
            logger.info("Cota de tokens insuficiente (%s < ~%s); aguardando %.0fs",
                        remaining, estimated, wait)
            time.sleep(wait)
            quota["tokens_remaining"] = quota.get("tokens_limit", estimated)
        calls.append(time.monotonic())

    def _invoke_groq(self, messages: list):
        """Lê os headers sem fazer uma segunda chamada só para consultar cotas."""
        client = getattr(self.llm, "client", None)
        raw_client = getattr(client, "with_raw_response", None)
        if raw_client is None:  # Permite clientes substitutos em testes.
            return self.llm.invoke(messages)
        groq_messages = [
            {"role": "system" if isinstance(m, SystemMessage) else "user",
             "content": m.content} for m in messages
        ]
        options = {}
        if self.model_name == "qwen/qwen3.8-27b" and self._output_budget() <= 1000:
            options["reasoning_effort"] = "none"
        raw = raw_client.create(
            model=self.model_name, messages=groq_messages,
            temperature=self.temperature, max_tokens=self._output_budget(), **options,
        )
        self._record_quota(raw.headers)
        completion = raw.parse()
        choice = completion.choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            raise GroqOutputTruncated(
                "Resposta truncada pelo teto de saída; dividir a extração antes de publicar."
            )
        return AIMessage(content=choice.message.content or "")

    def _invoke_with_rotation(self, messages: list, sleep_between: float = 8.0):
        """
        Rotaciona chaves ao esgotar a cota diária e tenta novamente falhas
        temporárias do serviço, sempre com um número finito de tentativas.
        """
        rate_limit_failures = 0
        transient_failures = 0
        failures_by_account = {}
        while True:
            try:
                if self._independent_accounts():
                    self._select_available_account(messages)
                    rate_limit_failures = failures_by_account.get(self._current_key_idx, 0)
                self._before_groq_request(messages, enforce_rpm=(
                    self._independent_accounts() or rate_limit_failures == 0))
                return self._invoke_groq(messages)
            except Exception as e:
                self._record_quota(getattr(getattr(e, "response", None), "headers", None))
                err_str = str(e)
                is_rate_limit = (getattr(e, "status_code", None) == 429
                                 or "429" in err_str or "rate_limit" in err_str.lower())
                is_daily_limit = "tokens per day" in err_str.lower() or "TPD" in err_str

                if is_rate_limit:
                    headers = getattr(getattr(e, "response", None), "headers", None) or {}
                    body = getattr(e, "body", None)
                    details = body.get("error", body) if isinstance(body, dict) else {}
                    if not isinstance(details, dict):
                        details = {}
                    message = str(details.get("message", "")) + " " + err_str
                    counts = {
                        label: re.search(rf"\b{label}\s*[:=]?\s*([\d,]+)", message, re.I)
                        for label in ("Limit", "Used", "Requested")
                    }
                    numbers = {label: int(match.group(1).replace(",", ""))
                               for label, match in counts.items() if match}
                    logger.warning(
                        "Groq 429: tipo=%s, código=%s, retry-after=%s; "
                        "tokens/min restantes=%s, requisições/dia restantes=%s",
                        details.get("type", "?"), details.get("code", "?"),
                        headers.get("retry-after", "?"),
                        headers.get("x-ratelimit-remaining-tokens", "?"),
                        headers.get("x-ratelimit-remaining-requests", "?"),
                    )
                    logger.warning("Detalhe Groq 429: %s", self._groq_error_detail(e))
                    if numbers:
                        logger.warning("Groq: limite=%s; usados=%s; solicitados=%s tokens",
                                       numbers.get("Limit", "?"), numbers.get("Used", "?"),
                                       numbers.get("Requested", "?"))
                    is_daily_limit = is_daily_limit or "tokens per day" in message.lower() or "tpd" in message.lower()
                    is_output_limit = ("output tokens per minute" in message.lower()
                                       or "otpm" in message.lower())
                    if is_output_limit and numbers.get("Limit", 0) > 0:
                        cap = max(1, int(numbers["Limit"] * 0.95))
                        if cap < self._output_budget():
                            self._output_token_cap = cap
                            self._rpm_budget_override = 1
                            # Output token budget only; no API credentials.
                            # nosemgrep: python.lang.security.audit.logging.logger-credential-leak.python-logger-credential-disclosure
                            logger.warning("Groq OTPM: reduzindo max_tokens para %s", cap)
                            continue
                    if not is_daily_limit and not is_output_limit and (
                        (numbers.get("Requested", 0) > numbers.get("Limit", float("inf")))
                        or "request too large" in message.lower()
                    ):
                        raise GroqRequestTooLarge(
                            "Groq recusou o tamanho deste pedido de tokens: "
                            + self._groq_error_detail(e)
                        ) from e
                    if is_daily_limit:
                        # Limite diário esgotado — tenta próxima chave
                        logger.warning(f"Chave {self._current_key_idx + 1} esgotou limite diário.")
                        if self._independent_accounts():
                            state = self._save_account()
                            state["quota"]["requests_remaining"] = 0
                            # Um limite diário de tokens também torna a conta indisponível.
                            state["quota"]["requests_reset_at"] = time.monotonic() + max(
                                301, self._rate_limit_wait(e))
                        elif not self._rotate_key():
                            raise RuntimeError(
                                "Groq: cota diária esgotada nas chaves configuradas; "
                                "a análise pode ser retomada a partir do checkpoint."
                            ) from e
                        rate_limit_failures = 0
                        transient_failures = 0
                        continue
                    else:
                        # O 429 pode indicar limite de tokens ou de requisições.
                        # Respeita o prazo informado pela API, sem assumir 60s.
                        self._rpm_budget_override = 1
                        if rate_limit_failures >= 3:
                            quota = getattr(self, "_quota", {})
                            if (not is_output_limit and details.get("type") == "tokens"
                                    and quota.get("tokens_limit")
                                    and quota.get("tokens_remaining", 0) >= quota["tokens_limit"]):
                                raise GroqRequestTooLarge(
                                    "Groq recusou este pedido mesmo com saldo de tokens/min "
                                    "renovado; tentando um lote menor."
                                ) from e
                            raise RuntimeError(
                                "Groq manteve o limite de uso (429) "
                                "após três esperas; retome após a cota ficar disponível. "
                                "O checkpoint continua salvo."
                            ) from e
                        wait_secs = self._rate_limit_wait(e)
                        if wait_secs > 300:
                            raise RuntimeError(
                                f"Groq pediu uma espera de {wait_secs:.0f}s por limite de uso "
                                "(429). Retome a análise depois desse prazo; "
                                "o checkpoint continua salvo."
                            ) from e
                        logger.warning("Groq: conta atual indisponível por %.0fs%s", wait_secs,
                                       "; verificando outras contas" if self._independent_accounts() else "")
                        rate_limit_failures += 1
                        if self._independent_accounts():
                            failures_by_account[self._current_key_idx] = rate_limit_failures
                            self._save_account()["cooldown_until"] = time.monotonic() + wait_secs
                        else:
                            time.sleep(wait_secs)
                        continue
                else:
                    status = getattr(e, "status_code", None)
                    transient = status in (500, 502, 503, 504) or (
                        status is None and (
                            "503" in err_str
                            or type(e).__name__ in {
                                "APIConnectionError", "APITimeoutError",
                                "ConnectError", "RemoteProtocolError", "TimeoutException",
                            }
                        )
                    )
                    if transient and transient_failures < 3:
                        delay = (5, 15, 30)[transient_failures]
                        transient_failures += 1
                        logger.warning(
                            "Falha temporária da Groq (%s). Tentativa adicional %s/3 em %ss",
                            status or type(e).__name__, transient_failures, delay,
                        )
                        time.sleep(delay)
                        continue
                    raise

    def analyze_chunk(self, chunk: dict, system_prompt: str | None = None) -> dict:
        """
        Analisa um único chunk de texto.

        Args:
            chunk: Chunk com conteúdo e metadados
            system_prompt: Prompt de sistema customizado

        Returns:
            Dicionário com análise do chunk
        """
        logger.info(f"Analisando chunk da seção: {chunk.get('section', 'unknown')}")

        if self.mock_mode:
            # Resposta simulada em modo mock
            mock_response = f"Análise mock do chunk: {chunk['content'][:50]}..."
            return {
                "requirements": self._extract_requirements_mock(chunk),
                "analysis": mock_response,
                "raw_response": mock_response,
                "chunk_metadata": {
                    "section": chunk.get("section"),
                    "content_length": len(chunk["content"]),
                },
                "system_prompt": system_prompt,
            }

        # Análise real com LLM
        prompt = self._build_analysis_prompt(chunk)

        if system_prompt:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt),
            ]
        else:
            messages = [HumanMessage(content=prompt)]

        response = self._invoke_with_rotation(messages)

        return {
            "requirements": self._extract_requirements_from_response(response.content),
            "analysis": response.content,
            "raw_response": response.content,
            "chunk_metadata": {
                "section": chunk.get("section"),
                "content_length": len(chunk["content"]),
            },
            "system_prompt": system_prompt,
        }

    def analyze_chunks(
        self, chunks: list[dict], system_prompt: str | None = None
    ) -> dict:
        """
        Analisa múltiplos chunks.

        Args:
            chunks: Lista de chunks
            system_prompt: Prompt de sistema customizado

        Returns:
            Dicionário com análises de todos os chunks
        """
        logger.info(f"Analisando {len(chunks)} chunks")

        if not chunks:
            return {"total_chunks": 0, "analyses": [], "system_prompt": system_prompt}

        analyses = []
        for chunk in chunks:
            analysis = self.analyze_chunk(chunk, system_prompt)
            analyses.append(analysis)

        return {
            "total_chunks": len(chunks),
            "analyses": analyses,
            "system_prompt": system_prompt,
        }

    def validate_analysis_output(self, output: dict) -> bool:
        """
        Valida a saída da análise.

        Args:
            output: Dicionário de saída da análise

        Returns:
            True se válido, False caso contrário
        """
        required_fields = ["requirements", "analysis", "raw_response"]

        for field in required_fields:
            if field not in output:
                logger.error(f"Campo obrigatório ausente: {field}")
                return False

        if not output["requirements"]:
            logger.warning("Nenhum requisito extraído")

        logger.info("Validação de análise concluída")
        return True

    def _build_analysis_prompt(self, chunk: dict) -> str:
        """Constrói o prompt para análise do chunk."""
        section = chunk.get("section", "desconhecida")
        content = chunk["content"]

        prompt = f"""
Analise o seguinte chunk de um edital de licitação:

Seção: {section}
Conteúdo: {content}

Identifique:
1. Requisitos técnicos mencionados
2. Prazos ou cronogramas
3. Documentação exigida
4. Pontos críticos ou ambíguos

Responda de forma estruturada e concisa.
"""
        return prompt

    def _extract_requirements_mock(self, chunk: dict) -> list[str]:
        """Extrai requisitos simulados em modo mock."""
        content = chunk["content"].lower()

        requirements = []

        # Simples extrações baseadas em palavras-chave
        if "processador" in content or "cpu" in content:
            requirements.append("Requisito de processador encontrado")
        if "memoria" in content or "ram" in content:
            requirements.append("Requisito de memória encontrado")
        if "prazo" in content:
            requirements.append("Prazo mencionado")
        if "document" in content:
            requirements.append("Documentação exigida")

        return requirements

    def _extract_requirements_from_response(self, response: str) -> list[str]:
        """Extrai requisitos da resposta do LLM."""
        # Implementação básica - pode ser melhorada com parsing mais sofisticado
        lines = response.split("\n")
        requirements = []

        for line in lines:
            line = line.strip()
            if line and (
                "requisito" in line.lower()
                or "exigência" in line.lower()
                or "deve" in line.lower()
            ):
                requirements.append(line)

        return requirements

    def extract_structured_info(self, chunk: dict) -> dict:
        """
        Extrai informações estruturadas de um chunk.

        Args:
            chunk: Chunk com conteúdo e metadados

        Returns:
            Dicionário com informações categorizadas
        """
        logger.info(
            f"Extraindo informações estruturadas da seção: {chunk.get('section', 'unknown')}"
        )

        content = chunk["content"].lower()
        section = chunk.get("section", "")

        result = {
            "technical_requirements": self._extract_technical_requirements(
                content, section
            ),
            "deadlines": self._extract_deadlines(content, section),
            "documentation": self._extract_documentation(content, section),
            "object_info": self._extract_object_info(content, section),
            "risk_factors": self._extract_risk_factors(content, section),
            "section": section,
        }

        logger.info(
            f"Informações extraídas: {len(result['technical_requirements'])} técnicos, {len(result['deadlines'])} prazos"
        )
        return result

    def validate_structured_output(self, output: dict) -> bool:
        """
        Valida a saída estruturada.

        Args:
            output: Dicionário de saída estruturada

        Returns:
            True se válido, False caso contrário
        """
        required_categories = [
            "technical_requirements",
            "deadlines",
            "documentation",
            "object_info",
            "risk_factors",
        ]

        for category in required_categories:
            if category not in output:
                logger.error(f"Categoria obrigatória ausente: {category}")
                return False

        # Verifica tipos
        list_categories = [
            "technical_requirements",
            "deadlines",
            "documentation",
            "risk_factors",
        ]
        for category in list_categories:
            if not isinstance(output[category], list):
                logger.error(f"Categoria {category} deve ser uma lista")
                return False

        # object_info pode ser dicionário
        if not isinstance(output["object_info"], dict):
            logger.error("object_info deve ser um dicionário")
            return False

        logger.info("Validação estruturada concluída")
        return True

    # ------------------------------------------------------------------
    # Helpers de limpeza de texto
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_sentences(sentences: list[str], min_words: int = 4, max_chars: int = 180) -> list[str]:
        """
        Remove frases muito curtas, muito longas ou claramente inúteis.

        Regras:
        - Pelo menos `min_words` palavras
        - No máximo `max_chars` caracteres (trunca com reticências)
        - Sem linhas que sejam só números/pontuação
        """
        seen: set[str] = set()
        result: list[str] = []
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            words = s.split()
            if len(words) < min_words:
                continue
            # Remove frases que são basicamente ruído (só dígitos/pontuação)
            if sum(c.isalpha() for c in s) < 10:
                continue
            # Trunca frases longas demais
            if len(s) > max_chars:
                s = s[:max_chars].rsplit(" ", 1)[0] + "..."
            # Dedup por conteúdo normalizado
            key = " ".join(words[:8])  # primeiras 8 palavras como chave
            if key in seen:
                continue
            seen.add(key)
            result.append(s)
        return result

    def _extract_technical_requirements(self, content: str, section: str) -> list[str]:
        """Extrai requisitos técnicos do conteúdo."""
        # Palavras-chave específicas o suficiente para não gerar ruído
        tech_keywords = [
            "processador", "cpu", "memória ram", "armazenamento",
            "ssd", "hdd", "monitor", "placa de vídeo",
            "sistema operacional", "especificação técnica",
            "norma abnt", "certificação iso", "certificação abnt",
        ]

        found: list[str] = []
        sentences = content.split(".")
        for keyword in tech_keywords:
            for sentence in sentences:
                if keyword in sentence:
                    found.append(sentence)

        return self._clean_sentences(found, min_words=5, max_chars=160)

    def _extract_deadlines(self, content: str, section: str) -> list[str]:
        """
        Extrai prazos usando regex para capturar só o trecho relevante
        (ex.: '30 dias', '10 dias úteis', '15/03/2026'), não a frase inteira.
        """
        import re

        patterns = [
            # "prazo de X [por extenso] dias [úteis/corridos] ..."
            r"prazo\s+de\s+(?:\d+|\w+)\s*(?:\([^)]+\)\s*)?\s*dias?[^,;\n.]{0,60}",
            # "X (por extenso) dias após/contado/a contar"
            r"\d+\s*(?:\([^)]+\)\s*)?dias?\s+[a-záàâãéêíóôõúç ]{0,15}(?:após|contado|a contar)[^,;\n.]{0,50}",
            # "até DD/MM/AAAA"
            r"até\s+\d{1,2}/\d{1,2}/\d{2,4}",
            # "vigência de X meses/anos"
            r"vigência\s+de\s+\d+\s*(?:meses?|anos?)[^,;\n.]{0,40}",
            # "entrega em/no prazo de X dias"
            r"entrega\s+(?:em|no\s+prazo\s+de)\s+\d+\s*dias?[^,;\n.]{0,40}",
        ]

        found: list[str] = []
        for pattern in patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                text = match.group(0).strip().rstrip(",;:")
                if len(text.split()) >= 3:  # mínimo 3 palavras
                    found.append(text)

        # Dedup por prefixo (primeiros 35 chars)
        seen: set[str] = set()
        result: list[str] = []
        for item in found:
            key = item[:35].lower()
            if key not in seen:
                seen.add(key)
                result.append(item.capitalize())

        return result[:20]

    def _extract_documentation(self, content: str, section: str) -> list[str]:
        """
        Extrai nomes de documentos, não frases inteiras.
        Prioriza a keyword em si + palavras adjacentes para nomear o documento.
        """
        import re

        # Documentos com nome fixo — retorna o nome canônico direto
        fixed_docs = {
            "certidão negativa de débitos": "Certidão Negativa de Débitos (CND)",
            "certidão negativa": "Certidão Negativa",
            "certidão positiva": "Certidão Positiva com Efeito de Negativa",
            "balanço patrimonial": "Balanço Patrimonial",
            "demonstrativo de resultados": "Demonstrativo de Resultados",
            "registro no cnpj": "Comprovante de Inscrição no CNPJ",
            "cnpj": "Cartão CNPJ",
            "contrato social": "Contrato Social",
            "estatuto social": "Estatuto Social",
            "ata de eleição": "Ata de Eleição da Diretoria",
            "procuração": "Procuração",
            "atestado de capacidade técnica": "Atestado de Capacidade Técnica",
            "atestado técnico": "Atestado Técnico",
            "licença de funcionamento": "Licença de Funcionamento",
            "alvará": "Alvará de Funcionamento",
            "cnh": "CNH",
            "registro geral": "RG",
            r"\brg\b": "RG",
            r"\bcpf\b": "CPF",
            "inscrição estadual": "Inscrição Estadual",
            "inscrição municipal": "Inscrição Municipal",
            "certidão trabalhista": "Certidão de Regularidade Trabalhista (CNDT)",
            "prova de regularidade": "Prova de Regularidade Fiscal",
            "comprovante de regularidade": "Comprovante de Regularidade",
            "declaração de idoneidade": "Declaração de Idoneidade",
            "declaração de inexistência": "Declaração de Inexistência de Fatos Impeditivos",
        }

        found: list[str] = []
        seen: set[str] = set()

        for pattern, canonical_name in fixed_docs.items():
            if re.search(pattern, content, re.IGNORECASE):
                if canonical_name not in seen:
                    seen.add(canonical_name)
                    found.append(canonical_name)

        return found

    def _extract_object_info(self, content: str, section: str) -> dict:
        """Extrai informações do objeto/bem."""
        info = {"description": "", "quantity": None, "unit": None}

        # Tenta extrair quantidade
        quantity_pattern = (
            r"(\d+)\s*(unidade|computadores?|itens?|equipamentos?|dispositivos?)"
        )
        match = re.search(quantity_pattern, content)
        if match:
            info["quantity"] = int(match.group(1))
            info["unit"] = match.group(2)

        # Tenta extrair descrição do objeto
        if "objeto" in content or "aquisição" in content:
            sentences = content.split(".")
            for sentence in sentences:
                if "objeto" in sentence or "aquisição" in sentence:
                    info["description"] = sentence.strip()
                    break

        return info

    def _extract_risk_factors(self, content: str, section: str) -> list[str]:
        """Extrai fatores de risco do conteúdo."""
        risks = []

        risk_keywords = [
            "risco",
            "crítico",
            "urgente",
            "complexo",
            "difícil",
            "curto prazo",
            "multa",
        ]

        for keyword in risk_keywords:
            if keyword in content:
                sentences = content.split(".")
                for sentence in sentences:
                    if keyword in sentence:
                        risks.append(sentence.strip())

        return risks

    def identify_critical_points(self, chunk: dict) -> dict:
        """
        Identifica pontos críticos no chunk.

        Args:
            chunk: Chunk com conteúdo e metadados

        Returns:
            Dicionário com pontos críticos e nível de risco
        """
        logger.info(
            f"Identificando pontos críticos na seção: {chunk.get('section', 'unknown')}"
        )

        content = chunk["content"].lower()
        section = chunk.get("section", "")

        if not content.strip():
            return {
                "critical_points": [],
                "risk_level": "BAIXO",
                "recommendations": [],
                "ambiguities": [],
            }

        # Identifica pontos críticos
        critical_points = self._extract_critical_keywords(content)

        # Classifica nível de risco
        risk_level = self._classify_risk_level(content, critical_points)

        # Gera recomendações
        recommendations = self._generate_recommendations(
            content, critical_points, risk_level
        )

        # Identifica ambiguidades
        ambiguities = self._identify_ambiguities(content)

        return {
            "critical_points": critical_points,
            "risk_level": risk_level,
            "recommendations": recommendations,
            "ambiguities": ambiguities,
            "section": section,
        }

    def summarize_critical_points(self, chunks: list[dict]) -> dict:
        """
        Resume pontos críticos de múltiplos chunks.

        Args:
            chunks: Lista de chunks

        Returns:
            Dicionário com resumo crítico
        """
        logger.info(f"Resumindo pontos críticos de {len(chunks)} chunks")

        all_critical_points = []
        all_risk_levels = []

        for chunk in chunks:
            analysis = self.identify_critical_points(chunk)
            all_critical_points.extend(analysis["critical_points"])
            all_risk_levels.append(analysis["risk_level"])

        # Determina nível de risco geral
        overall_risk = self._calculate_overall_risk(all_risk_levels)

        # Ações prioritárias
        priority_actions = self._generate_priority_actions(
            all_critical_points, overall_risk
        )

        return {
            "total_critical_points": len(all_critical_points),
            "overall_risk_level": overall_risk,
            "priority_actions": priority_actions,
            "critical_points_by_risk": self._group_by_risk(
                all_critical_points, all_risk_levels
            ),
        }

    def validate_critical_analysis(self, output: dict) -> bool:
        """
        Valida a análise crítica.

        Args:
            output: Dicionário de saída da análise crítica

        Returns:
            True se válido, False caso contrário
        """
        required_fields = ["critical_points", "risk_level", "recommendations"]

        for field in required_fields:
            if field not in output:
                logger.error(f"Campo obrigatório ausente: {field}")
                return False

        # Verifica se risk_level é válido
        valid_risk_levels = ["ALTO", "MEDIO", "BAIXO"]
        if output["risk_level"] not in valid_risk_levels:
            logger.error(f"Nível de risco inválido: {output['risk_level']}")
            return False

        logger.info("Validação crítica concluída")
        return True

    def _extract_critical_keywords(self, content: str) -> list[str]:
        """
        Extrai pontos críticos do conteúdo.

        Filtra frases que claramente vieram corrompidas do PDF:
        - Muitas palavras com ≤2 letras (abreviações/ruído)
        - Menos de 6 palavras reais
        - Referências ao nome do edital (repetitivas, sem valor)
        """
        critical_keywords = [
            "impossível", "inviável", "curto prazo", "multa", "penalidade",
            "severo", "obrigatório", "exclusivo", "único", "imediato",
            "urgente", "complexo", "difícil", "risco", "perigo",
            "impedimento", "sanção", "rescisão", "inadimplência",
        ]

        # Padrões de ruído que indicam texto corrompido
        noise_patterns = [
            r"\bedital\s+concorrência\s+eletrônica\b",  # referência repetitiva ao edital
            r"\bprocesso\s+administrativo\s+n[°º]",
            r"[a-z]\)\s+[a-z]\)\s+[a-z]\)",             # sequências tipo "a) b) c)"
        ]

        candidates: list[str] = []
        sentences = content.split(".")
        for keyword in critical_keywords:
            for sentence in sentences:
                if keyword in sentence:
                    candidates.append(sentence)

        result: list[str] = []
        seen: set[str] = set()

        for sent in candidates:
            sent = sent.strip()
            if not sent:
                continue

            # Descarta se contém padrões de ruído
            is_noisy = any(re.search(p, sent, re.IGNORECASE) for p in noise_patterns)
            if is_noisy:
                continue

            words = sent.split()
            # Descarta frases muito curtas
            if len(words) < 6:
                continue

            # Descarta se >40% das palavras têm ≤2 letras (texto corrompido)
            short_words = sum(1 for w in words if len(re.sub(r"[^a-záàâãéêíóôõúç]", "", w.lower())) <= 2)
            if short_words / len(words) > 0.40:
                continue

            # Trunca e dedup
            if len(sent) > 200:
                sent = sent[:200].rsplit(" ", 1)[0] + "..."
            key = " ".join(words[:7])
            if key in seen:
                continue
            seen.add(key)
            result.append(sent.capitalize())

        return result[:15]  # máximo 15 pontos críticos por chunk

    def _classify_risk_level(self, content: str, critical_points: list[str]) -> str:
        """Classifica o nível de risco."""
        if len(critical_points) >= 3:
            return "ALTO"
        elif len(critical_points) >= 1:
            return "MEDIO"
        else:
            return "BAIXO"

    def _generate_recommendations(
        self, content: str, critical_points: list[str], risk_level: str
    ) -> list[str]:
        """Gera recomendações baseadas nos pontos críticos."""
        recommendations = []

        if "prazo" in content:
            recommendations.append("Verificar viabilidade do prazo estabelecido")

        if "multa" in content or "penalidade" in content:
            recommendations.append("Avaliar impacto financeiro das penalidades")

        if "obrigatório" in content or "exclusivo" in content:
            recommendations.append(
                "Confirmar capacidade de atender requisitos obrigatórios"
            )

        if risk_level == "ALTO":
            recommendations.append("Recomendada revisão detalhada do edital")

        return recommendations

    def _identify_ambiguities(self, content: str) -> list[str]:
        """Identifica ambiguidades no conteúdo."""
        ambiguity_keywords = [
            "a ser definido",
            "posteriormente",
            "conforme",
            "apropriado",
            "adequado",
            "suficiente",
            "necessário",
            "de acordo com",
        ]

        ambiguities = []
        for keyword in ambiguity_keywords:
            if keyword in content:
                sentences = content.split(".")
                for sentence in sentences:
                    if keyword in sentence:
                        ambiguities.append(sentence.strip())

        return ambiguities

    def _calculate_overall_risk(self, risk_levels: list[str]) -> str:
        """Calcula nível de risco geral."""
        if "ALTO" in risk_levels:
            return "ALTO"
        elif "MEDIO" in risk_levels:
            return "MEDIO"
        else:
            return "BAIXO"

    def _generate_priority_actions(
        self, critical_points: list[str], overall_risk: str
    ) -> list[str]:
        """Gera ações prioritárias."""
        actions = []

        if overall_risk == "ALTO":
            actions.append("Reunião imediata com equipe técnica")
            actions.append("Análise detalhada de capacidade")
        elif overall_risk == "MEDIO":
            actions.append("Revisão de requisitos")
            actions.append("Planejamento de contingência")
        else:
            actions.append("Acompanhamento padrão do processo")

        return actions

    def _group_by_risk(
        self, critical_points: list[str], risk_levels: list[str]
    ) -> dict:
        """Agrupa pontos críticos por nível de risco."""
        grouped = {"ALTO": [], "MEDIO": [], "BAIXO": []}

        for point, risk in zip(critical_points, risk_levels):
            grouped[risk].append(point)

        return grouped

    def _build_structured_prompt(self, chunks: list[dict]) -> str:
        """
        Monta prompt para análise estruturada de múltiplos chunks via LLM.
        Retorna JSON com campos padronizados.
        """
        # Concatena conteúdo dos chunks com identificação de seção
        sections_text = ""
        for chunk in chunks:
            section = chunk.get("section", "geral")
            content = chunk.get("content", "").strip()
            if content:
                sections_text += f"\n--- SEÇÃO: {section.upper()} ---\n{content}\n"

        prompt = f"""Você é um especialista em licitações públicas brasileiras (Lei 14.133/2021).
Analise os trechos abaixo de um edital de licitação e extraia as informações solicitadas.

TRECHOS DO EDITAL:
{sections_text}

Responda SOMENTE com um JSON válido no seguinte formato (sem markdown, sem explicações fora do JSON):
{{
  "objeto": "descrição resumida do objeto da licitação em 1-2 frases",
  "requisitos_tecnicos": ["lista de requisitos técnicos objetivos, cada item em 1 frase clara"],
  "documentos_exigidos": ["nome canônico de cada documento exigido, ex: Certidão Negativa de Débitos, CNPJ, Atestado Técnico"],
  "prazos": ["cada prazo em formato legível, ex: 30 dias úteis para entrega"],
  "pontos_criticos": ["cada ponto crítico ou risco em 1 frase objetiva"],
  "nivel_risco": "ALTO|MEDIO|BAIXO",
  "recomendacoes": ["até 3 recomendações objetivas para a empresa participante"]
}}

Regras:
- Seja objetivo e conciso. Cada item deve ser uma frase completa e compreensível.
- Não repita informações entre campos.
- Se não houver informação para um campo, use lista vazia [].
- Para nivel_risco: ALTO se houver multas severas/prazos curtos/requisitos complexos, MEDIO se moderado, BAIXO se simples.
"""
        return prompt

    def _parse_llm_json(self, response: str) -> dict:
        """
        Parseia o JSON retornado pelo LLM de forma robusta.
        Tenta extrair o JSON mesmo se o LLM incluir texto extra.
        """
        import json

        # Tenta parsear direto
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass

        # Tenta extrair bloco JSON com regex
        match = re.search(r'\{[\s\S]*\}', response)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # Fallback: retorna estrutura vazia
        logger.warning("Não foi possível parsear JSON do LLM, usando fallback")
        return {
            "objeto": "",
            "requisitos_tecnicos": [],
            "documentos_exigidos": [],
            "prazos": [],
            "pontos_criticos": [],
            "nivel_risco": "BAIXO",
            "recomendacoes": [],
        }

    # ──────────────────────────────────────────────────────────────
    # Queries RAG temáticas
    # ──────────────────────────────────────────────────────────────

    # Cada entrada: (chave_interna, label_display, texto_da_query_para_embedding)
    RAG_QUERIES: list[tuple[str, str, str]] = [
        (
            "objeto_licitacao",
            "Objeto da Licitação",
            "objeto da licitação contratação aquisição serviço bem fornecimento "
            "descrição do objeto finalidade do contrato o que será contratado",
        ),
        (
            "posso_participar",
            "Posso participar?",
            "requisitos habilitação participação licitante empresa qualificação "
            "escolaridade idade registro profissional documentos específicos "
            "impedimento inabilitação condições de participação",
        ),
        (
            "prazos",
            "Quais são os prazos?",
            "prazo entrega vigência contrato cronograma data limite "
            "dias úteis corridos início execução encerramento",
        ),
        (
            "custos_pagamento",
            "Quanto custa e como pago?",
            "valor estimado preço pagamento faturamento nota fiscal "
            "reajuste reequilíbrio financeiro garantia caução depósito",
        ),
        (
            "selecao_processo",
            "Como será a seleção ou avaliação?",
            "critério julgamento menor preço técnica proposta avaliação "
            "pontuação classificação habilitação etapas processo seleção",
        ),
        (
            "objeto_escopo",
            "O que devo entregar ou produzir?",
            "objeto escopo serviço produto bem especificação técnica "
            "entregável quantidade unidade memorial descritivo planilha",
        ),
        (
            "eliminacao_penalidades",
            "Quais são as regras de eliminação?",
            "eliminação desclassificação inabilitação penalidade multa "
            "sanção rescisão impedimento declaração inidoneidade",
        ),
        (
            "documentos",
            "Quais documentos são exigidos?",
            "documentação exigida certidão comprovante declaração atestado "
            "registro alvará licença CNPJ balanço contrato social habilitação "
            "regularidade fiscal trabalhista jurídica técnica",
        ),
        (
            "pontos_criticos",
            "Pontos críticos e riscos",
            "risco crítico urgente multa penalidade obrigatório prazo curto "
            "complexo impossível severo rescisão inadimplência garantia",
        ),
    ]

    def _retrieve_rag_chunks(
        self,
        top_k: int = 5,
        keys_override: list[str] | None = None,
    ) -> dict[str, list[dict]]:
        """
        Para cada query RAG temática, busca os top_k chunks mais relevantes
        no ChromaDB usando similaridade vetorial.

        Args:
            top_k: Número de chunks a recuperar por query
            keys_override: Se fornecido, busca apenas as queries com essas chaves

        Returns:
            Dict: {chave_query: [chunks_relevantes]}
        """
        from .node_2_embeddings import Node2EmbeddingGenerator

        logger.info(f"RAG: iniciando busca vetorial em '{self.persist_directory}'")

        node2 = Node2EmbeddingGenerator()
        results_by_query: dict[str, list[dict]] = {}

        queries_to_run = [
            (key, label, qt) for key, label, qt in self.RAG_QUERIES
            if keys_override is None or key in keys_override
        ]

        for key, label, query_text in queries_to_run:
            try:
                raw = node2.search_similar(
                    query=query_text,
                    collection_name=self.collection_name,
                    n_results=top_k,
                    persist_directory=self.persist_directory,
                )
                docs = raw.get("documents", [[]])[0]
                metas = raw.get("metadatas", [[]])[0]

                chunks = [
                    {
                        "content": doc,
                        "section": meta.get("section", "geral"),
                        "metadata": meta,
                        "rag_query": label,
                    }
                    for doc, meta in zip(docs, metas)
                    if doc and doc.strip()
                ]
                results_by_query[key] = chunks
                logger.info(f"RAG '{label}': {len(chunks)} chunks recuperados")

            except Exception as e:
                raise RuntimeError(f"Falha na busca vetorial RAG para '{label}': {e}") from e

        return results_by_query


    def _build_unified_prompt(self, chunks: list[dict], preceding: list[dict] | None = None) -> str:
        """
        Prompt unificado: extrai TODAS as informacoes relevantes de um batch
        em uma unica chamada ao LLM.
        """
        context = "\n\n".join(
            f"[Trecho {i+1}, página {c.get('metadata', {}).get('page', 'não informada')}]\n{c['content'].strip()}"
            for i, c in enumerate(chunks)
            if c.get("content", "").strip()
        )
        parent_context = "\n".join(
            f"[Contexto anterior, página {c.get('metadata', {}).get('page', 'não informada')}]\n"
            f"{c['content'][-900:]}" for c in (preceding or []) if c.get("content")
        )
        return f"""Especialista em licitacoes publicas brasileiras (Lei 14.133/2021).

Analise os trechos abaixo e extraia TODAS as informacoes encontradas.

TRECHOS:
{context}

CONTEXTO ANTERIOR (apenas para verificar condições; não extraia novas regras daqui):
{parent_context}

Responda SOMENTE com JSON valido (sem markdown):
{{
  "objeto": "descricao do objeto se encontrado, senao string vazia",
  "documentos": ["documentos que o licitante deve apresentar na proposta ou habilitacao, ex: certidao, atestado ou planilha"],
  "documentos_execucao": ["documentos a produzir ou entregar somente apos a contratacao, se a etapa estiver explicita"],
  "anexos_referencia": ["anexos e documentos fornecidos pelo orgao para consulta, sem pedido de entrega pelo licitante"],
  "pendencias_documentais": ["documentos citados cuja etapa ou exigencia de entrega nao ficou clara nos trechos"],
  "regras": [{{"tipo":"participacao ou selecao", "titulo":"regra completa e fiel ao texto", "trecho_id":1, "citacao":"frase literal da cláusula", "condicao":"condição literal que limita a regra, ou string vazia"}}],
  "prazos": ["cada prazo com valor, ex: 90 dias corridos para execucao, 30 dias para pagamento"],
  "custos": ["valores e formas de pagamento encontrados"],
  "entregas": ["o que deve ser entregue ou executado"],
  "eliminacao": ["causas de eliminacao ou penalidades com valores"],
  "riscos": ["riscos e pontos criticos objetivos"],
  "nivel_risco": "ALTO|MEDIO|BAIXO"
}}

Regras:
- Inclua apenas o que esta EXPLICITAMENTE nos trechos.
- Classifique cada documento em UMA das quatro listas: entrega na proposta/habilitacao,
  entrega apos contratacao, anexo fornecido pelo orgao ou etapa incerta.
- Projeto Basico, minutas e anexos para consulta nao sao documentos do checklist.
- Condicoes, proibicoes e atividades vao em regras, eliminacao ou entregas.
- Nao crie um documento a partir de uma obrigacao que nao pede comprovante explicito.
- Em regras, tipo selecao abrange disputa, julgamento e avaliacao de propostas;
  participacao abrange condicoes de participacao e habilitacao. Obrigações
  exclusivas da execução contratual vão em entregas, não em participacao.
- Para CADA regra, informe trecho_id e uma citacao literal do próprio trecho.
  Use uma frase curta da cláusula (até 350 caracteres), preservando seu sentido.
  O titulo deve preservar sujeito, condição e consequência, sem inverter a
  relação entre valores. Não extraia um título de uma frase subordinada sem
  considerar o início da cláusula; quando faltar contexto, não crie a regra.
- Se a regra depender de "caso", "se", "quando", "desde que" ou de uma
  modalidade alternativa, copie a condição literal em condicao. A condição
  deve existir em algum trecho desta mesma página no lote ou no contexto
  anterior. Não a trate como
  aplicável apenas porque está descrita no documento.
- Nao repita um mesmo fato com palavras diferentes no mesmo campo.
- Se trechos parecerem contraditorios, preserve as formulacoes sem escolher uma como correta.
- Se nao ha informacao para um campo, use lista vazia ou string vazia.
- Seja conciso: cada item em no maximo 15 palavras.
"""

    @staticmethod
    def _verify_extracted_rules(rows: list, batch: list[dict],
                                preceding: list[dict] | None = None) -> dict[str, list[dict]]:
        """Liga cada regra à página e a texto literal do chunk informado."""
        if not isinstance(rows, list):
            raise ValueError("Regras sem fontes: lista 'regras' ausente")

        def literal(value: str) -> str:
            return " ".join(value.split()).casefold()

        sources = {"participacao": [], "selecao": []}
        for row in rows:
            if not isinstance(row, dict) or row.get("tipo") not in sources:
                raise ValueError("Regra sem tipo válido")
            title, quote, condition = (row.get("titulo"), row.get("citacao"),
                                       row.get("condicao", ""))
            chunk_index = row.get("trecho_id")
            if (not isinstance(title, str) or len(title.strip()) < 5
                    or not isinstance(quote, str) or not 15 <= len(quote.strip()) <= 750
                    or not isinstance(condition, str)
                    or type(chunk_index) is not int or not 1 <= chunk_index <= len(batch)):
                raise ValueError("Regra sem título, citação ou trecho de origem válido")
            source_chunk = batch[chunk_index - 1]
            metadata = source_chunk.get("metadata", {})
            page = metadata.get("page")
            # Reconstrói somente a fronteira contígua de um subtrecho. Assim,
            # uma citação/condição que começa na parte anterior pode ser validada
            # sem inventar texto nem buscar evidência em outra página.
            start = metadata.get("source_char_start")
            if type(start) is int and start > 0:
                prefix = next((chunk for chunk in reversed(preceding or [])
                               if chunk.get("metadata", {}).get("page") == page
                               and chunk.get("metadata", {}).get("chunk_id") == metadata.get("chunk_id")
                               and chunk.get("metadata", {}).get("source_char_end") == start), None)
                if prefix is not None:
                    source_chunk = {**source_chunk,
                                    "content": prefix["content"] + source_chunk["content"]}
            if page is None:
                raise ValueError("Citação da regra não aparece no trecho e página informados")
            if literal(quote) not in literal(source_chunk.get("content", "")):
                # Um trecho_id incorreto não precisa descartar todo o lote se
                # a citação literal estiver em outro chunk da mesma página.
                matching = [
                    candidate for candidate in batch + (preceding or [])
                    if candidate.get("metadata", {}).get("page") == page
                    and literal(quote) in literal(candidate.get("content", ""))
                ]
                if not matching:
                    raise ValueError("Citação da regra não aparece no trecho e página informados")
                source_chunk = matching[0]
            condition_supported = not condition or any(
                c.get("metadata", {}).get("page") == page
                and literal(condition) in literal(c.get("content", ""))
                for c in [source_chunk] + batch + (preceding or [])
            )
            # A citação já comprovada pode ser a condição literal. Isso evita
            # rejeitar a regra apenas porque o modelo parafraseou esse campo.
            conditional_quote = re.search(
                r"(?i)(?:^|[.;]\s*)\s*(?:\d+(?:\.\d+)*\.?\s*)?"
                r"(?:caso|quando|desde que|na hipótese de)\b", quote.strip()
            )
            if condition and not condition_supported and conditional_quote:
                condition = quote
                condition_supported = True
            if not condition_supported:
                raise ValueError("Condição da regra não aparece na página informada")
            if not condition and conditional_quote:
                condition = quote
            sources[row["tipo"]].append({
                "item": title.strip(), "pagina": page, "citacao": quote.strip(),
                "condicao": condition.strip(),
                "chunk_id": source_chunk.get("metadata", {}).get("chunk_id"),
            })
        return sources

    def _extract_verified_batch(self, batch: list[dict], system_msg,
                                preceding: list[dict] | None = None) -> tuple[dict, dict]:
        """Preserva regras comprovadas e separa candidatas sem citação verificável."""
        fields = ("documentos", "documentos_execucao", "anexos_referencia",
                  "pendencias_documentais", "prazos", "custos", "entregas",
                  "eliminacao", "riscos")
        prompt = self._build_unified_prompt(batch, preceding)
        correction = ""
        verified = {"participacao": [], "selecao": []}
        pending = {}
        last_parsed = None
        last_error = None
        for attempt in range(3):
            response = self._invoke_with_rotation([
                system_msg, HumanMessage(content=prompt + correction)
            ])
            try:
                parsed = self._parse_llm_json(response.content)
                if any(not isinstance(parsed.get(field), list) or
                       any(not isinstance(item, str) for item in parsed[field])
                       for field in fields):
                    raise ValueError("JSON de extração incompleto ou inválido")
                rows = parsed.get("regras")
                if not isinstance(rows, list):
                    raise ValueError("Regras sem fontes: lista 'regras' ausente")
                last_parsed = parsed
                errors = []
                for row in rows:
                    try:
                        checked = self._verify_extracted_rules([row], batch, preceding)
                        for topic, items in checked.items():
                            for item in items:
                                if item not in verified[topic]:
                                    verified[topic].append(item)
                                pending.pop((topic, item["item"].casefold()), None)
                    except ValueError as error:
                        title = row.get("titulo") if isinstance(row, dict) else None
                        topic = row.get("tipo") if isinstance(row, dict) else None
                        chunk_index = row.get("trecho_id") if isinstance(row, dict) else None
                        page = (batch[chunk_index - 1].get("metadata", {}).get("page")
                                if type(chunk_index) is int and 1 <= chunk_index <= len(batch)
                                else None)
                        candidate = {
                            "titulo_proposto": title if isinstance(title, str) else "Regra sem título",
                            "tipo_proposto": topic if topic in verified else "indefinido",
                            "pagina_sugerida": page,
                            "citacao_proposta": (row.get("citacao", "")[:750]
                                                  if isinstance(row, dict)
                                                  and isinstance(row.get("citacao"), str) else ""),
                            "motivo": str(error),
                        }
                        pending[(candidate["tipo_proposto"],
                                 candidate["titulo_proposto"].casefold())] = candidate
                        errors.append(str(error))
                if not errors and not pending:
                    return parsed, verified
                last_error = ValueError(errors[0] if errors else
                                        "Regra de tentativa anterior continua sem fonte")
            except (ValueError, json.JSONDecodeError) as error:
                last_error = error
            if attempt < 2:
                logger.warning("Fonte ou JSON inválido no lote (%s); repetindo resposta (%s/3)",
                               last_error, attempt + 2)
                correction = (
                    "\n\nA resposta anterior foi recusada: " + str(last_error) + ". "
                    "Refaça o JSON completo. Para cada regra, copie citacao e condicao "
                    "literalmente dos trechos fornecidos, na mesma página do trecho_id. "
                    "Se a condição não estiver nesses trechos, omita a regra dependente "
                    "dela. Não transforme regra condicional em incondicional. "
                    "Não altere nem invente números de página."
                )
                time.sleep(8)
        if last_parsed is not None:
            last_parsed["_unverified_rules"] = list(pending.values())
            if pending:
                logger.warning("Lote com %s regra(s) sem fonte literal; revisão sinalizada no relatório",
                               len(pending))
            return last_parsed, verified
        raise last_error or RuntimeError("Lote sem JSON válido")

    @staticmethod
    def _extraction_text_cut(text: str) -> int | None:
        """Prefere fronteiras de frases; não elimina caracteres entre partes."""
        if len(text) <= 80:
            return None
        lower, upper = len(text) // 3, len(text) * 2 // 3
        middle = len(text) // 2
        for pattern in (r"\n\s*\n", r"[.;!?]\s+", r"\s+"):
            boundaries = [match.end() for match in re.finditer(pattern, text)
                          if lower <= match.end() <= upper]
            if boundaries:
                return min(boundaries, key=lambda point: abs(point - middle))
        return middle

    def _extract_budgeted_batch(self, batch: list[dict], system_msg,
                                preceding: list[dict], checkpoint: dict,
                                batch_number: int) -> tuple[dict, dict]:
        """Divide pedidos grandes e salva cada parte antes de continuar."""
        cache = checkpoint.setdefault("partial_batches", {}).setdefault(str(batch_number), {})

        def merge(left, right):
            parsed = {}
            for key in set(left[0]) | set(right[0]):
                a, b = left[0].get(key), right[0].get(key)
                if isinstance(a, list) or isinstance(b, list):
                    parsed[key] = []
                    for item in (a or []) + (b or []):
                        if item not in parsed[key]:
                            parsed[key].append(item)
                elif key == "nivel_risco":
                    levels = {"BAIXO": 0, "MEDIO": 1, "ALTO": 2}
                    parsed[key] = max((a or "BAIXO", b or "BAIXO"),
                                      key=lambda value: levels.get(value, 0))
                else:
                    parsed[key] = a or b
            sources = {topic: [] for topic in ("participacao", "selecao")}
            for topic in sources:
                for item in left[1][topic] + right[1][topic]:
                    if item not in sources[topic]:
                        sources[topic].append(item)
            return parsed, sources

        def walk_text(index, lo, hi, context, known_error=None, depth=0):
            # Os offsets são relativos ao chunk original e integram a chave do
            # checkpoint. Mudar o tamanho da resposta não invalida partes salvas.
            key = f"{index}:{index + 1}/text:{lo}:{hi}"
            saved = cache.get(key, {})
            if "parsed" in saved and "sources" in saved:
                logger.info("Lote %s: reutilizando subtrecho %s salvo", batch_number, key)
                return saved["parsed"], saved["sources"]
            original = batch[index]
            text = original["content"][lo:hi]
            if "split_at" not in saved:
                error = known_error
                if error is None:
                    fragment = {**original, "content": text,
                                "metadata": {**original.get("metadata", {}),
                                             "source_char_start": lo, "source_char_end": hi}}
                    fragment_context = list(context)
                    if lo:
                        fragment_context.append({
                            **original, "content": original["content"][max(0, lo - 900):lo],
                            "metadata": {**original.get("metadata", {}),
                                         "source_char_start": max(0, lo - 900),
                                         "source_char_end": lo},
                        })
                    try:
                        result = self._extract_verified_batch([fragment], system_msg, fragment_context)
                    except GroqRequestTooLarge as failure:
                        error = failure
                    else:
                        cache[key] = {"parsed": result[0], "sources": result[1]}
                        self._save_checkpoint(checkpoint)
                        return result
                cut = self._extraction_text_cut(text)
                if cut is None or depth >= 8:
                    raise GroqRequestTooLarge(
                        f"Não foi possível concluir a extração de um subtrecho de {len(text)} "
                        "caracteres dentro do orçamento de tokens. As partes concluídas "
                        "continuam salvas no checkpoint. " + str(error)
                    ) from error
                saved = cache[key] = {"split_at": lo + cut}
                self._save_checkpoint(checkpoint)
                logger.warning("Lote %s: subdividindo texto do trecho %s, página %s, "
                               "caracteres %s:%s (resposta não coube no orçamento)",
                               batch_number, original.get("metadata", {}).get("chunk_id", index),
                               original.get("metadata", {}).get("page", "?"), lo, hi)
            cut = saved["split_at"]
            return merge(walk_text(index, lo, cut, context, depth=depth + 1),
                         walk_text(index, cut, hi, context, depth=depth + 1))

        def walk(start, end):
            key = f"{start}:{end}"
            saved = cache.get(key, {})
            if "parsed" in saved and "sources" in saved:
                logger.info("Lote %s: reutilizando parte %s salva", batch_number, key)
                return saved["parsed"], saved["sources"]
            context = preceding
            if start or end != len(batch):
                page = batch[start].get("metadata", {}).get("page")
                context = [chunk for chunk in preceding + batch[:start]
                           if chunk.get("metadata", {}).get("page") == page][-2:]
            if saved.get("text_split"):
                return walk_text(start, 0, len(batch[start]["content"]), context)
            if not saved.get("split"):
                try:
                    result = self._extract_verified_batch(batch[start:end], system_msg, context)
                except GroqRequestTooLarge as error:
                    if end - start <= 1:
                        cache[key] = {"text_split": True}
                        self._save_checkpoint(checkpoint)
                        return walk_text(start, 0, len(batch[start]["content"]), context,
                                         known_error=error)
                    cache[key] = {"split": True}
                    self._save_checkpoint(checkpoint)
                    logger.warning("Lote %s: dividindo parte %s para caber na cota de tokens",
                                   batch_number, key)
                else:
                    cache[key] = {"parsed": result[0], "sources": result[1]}
                    self._save_checkpoint(checkpoint)
                    return result
            middle = start + (end - start) // 2
            return merge(walk(start, middle), walk(middle, end))

        return walk(0, len(batch))

    @staticmethod
    def _report_context(agg: dict, limit_per_topic: int = 12) -> tuple[str, dict]:
        """Amostra distribuída dos itens extraídos, com cobertura explícita."""
        import json

        topics = {
            "participacao": agg["requisitos_participacao"],
            "criterios_tecnicos_e_selecao": agg["selecao"],
            "prazos": agg["prazos"],
            "custos": agg["custos"],
            "entregas": agg["entregas"],
            "eliminacao": agg["eliminacao"],
            "riscos": agg["riscos"],
        }
        coverage = {}
        excerpts = {}
        for topic, items in topics.items():
            count = min(len(items), limit_per_topic)
            if count == len(items):
                selected = items
            elif count == 1:
                selected = [items[0]]
            else:
                positions = [round(i * (len(items) - 1) / (count - 1)) for i in range(count)]
                selected = [items[pos] for pos in positions]
            excerpts[topic] = selected
            coverage[topic] = {"included": count, "total": len(items)}

        context = json.dumps(
            {"objeto": agg["objeto"], "nivel_risco": agg["nivel_risco"], "itens": excerpts},
            ensure_ascii=False,
        )
        return context, coverage

    def _synthesize_explanatory_report(
        self, agg: dict, source_refs: dict | None = None
    ) -> dict:
        """Gera explicação separada dos itens brutos, sem alterar o checklist."""
        context, coverage = self._report_context(agg)
        import json
        sources = source_refs or {}
        # Limita o contexto de evidências sem esconder as fontes completas,
        # que permanecem na resposta para conferência na interface.
        evidence = {
            topic: [
                {"page": item["page"], "excerpt": item["excerpt"][:350]}
                for item in items[:2]
            ]
            for topic, items in sources.items()
        }
        prompt = (
            "Escreva em português um relatório claro para quem está lendo um edital. "
            "Use SOMENTE os itens extraídos abaixo; não invente obrigações, valores, "
            "prazos, conclusões jurídicas nem aptidão da empresa. "
            "Se não houver informação, diga que não foi identificada na amostra. "
            "A amostra não representa necessariamente o edital inteiro. "
            "Responda APENAS em JSON com as chaves resumo e requisitos. "
            "Em resumo, escreva 2 parágrafos explicando o objeto, o que se espera "
            "do participante, prazos e principais pontos de atenção. "
            "Cada parágrafo deve ter no máximo 65 palavras. "
            "Em requisitos, escreva 2 ou 3 parágrafos conectando condições de participação, "
            "critérios técnicos e de seleção com suas implicações práticas, conforme os itens. "
            "Não faça lista e não inclua documentos obrigatórios: eles têm checklist próprio. "
            "Não repita contagens nem trate risco como decisão definitiva. "
            "Quando mencionar fato apoiado pelos trechos de evidência, cite [p. N] "
            "com a página física indicada; não crie páginas não fornecidas. "
            "Não atribua ao PDF fatos que apareçam apenas nos itens extraídos.\n\n"
            f"ITENS EXTRAÍDOS:\n{context}\n\n"
            f"TRECHOS RECUPERADOS DO MESMO EDITAL:\n{json.dumps(evidence, ensure_ascii=False)}"
        )
        response = self._invoke_with_rotation([
            SystemMessage(content="Redija apenas JSON válido, sem markdown."),
            HumanMessage(content=prompt),
        ])
        parsed = self._parse_llm_json(response.content)
        resumo = parsed.get("resumo")
        requisitos = parsed.get("requisitos")
        if not isinstance(resumo, str) or not resumo.strip():
            raise ValueError("Síntese do resumo ausente ou inválida")
        if not isinstance(requisitos, str) or not requisitos.strip():
            raise ValueError("Síntese dos requisitos ausente ou inválida")
        pages = {
            str(source["page"])
            for items in evidence.values() for source in items
            if source.get("page") is not None
        }
        for cited_page in re.findall(r"\[p\.\s*(\d+)\]", resumo + " " + requisitos):
            if cited_page not in pages:
                raise ValueError(f"A síntese citou uma página não recuperada: {cited_page}")
        return {
            "resumo": resumo.strip(),
            "requisitos": requisitos.strip(),
            "coverage": coverage,
        }

    def _load_checkpoint(self, chunks: list[dict], total_batches: int, empty_agg: dict) -> dict:
        """Carrega somente um checkpoint compatível com o conteúdo e a versão atuais."""
        payload = json.dumps(
            {
                "version": self.CHECKPOINT_VERSION,
                "model": getattr(self, "model_name", ""),
                "chunks": [
                    (chunk.get("content"), chunk.get("section"),
                     chunk.get("metadata", {}).get("page"))
                    for chunk in chunks
                ],
            }, ensure_ascii=False, sort_keys=True,
        )
        fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        directory = getattr(self, "checkpoint_directory", None)
        self._checkpoint_file = Path(directory) / f"{fingerprint}.json" if directory else None
        fresh = {
            "version": self.CHECKPOINT_VERSION, "fingerprint": fingerprint,
            "next_batch": 0, "agg": empty_agg, "max_risk": "BAIXO",
            "rule_sources": {"participacao": [], "selecao": []},
            "unverified_rules": [],
            "partial_batches": {},
            "explanation_version": self.EXPLANATION_VERSION,
            "explanations": {"participacao": {}, "selecao": {}},
        }
        if not self._checkpoint_file or not self._checkpoint_file.exists():
            if directory:
                legacy_payload = json.loads(payload)
                legacy_payload["version"] = 1
                old_fingerprint = hashlib.sha256(json.dumps(
                    legacy_payload, ensure_ascii=False, sort_keys=True,
                ).encode("utf-8")).hexdigest()
                if (Path(directory) / f"{old_fingerprint}.json").exists():
                    logger.info(
                        "Checkpoint anterior preservado; reextraindo regras para registrar "
                        "cláusula e página no novo formato"
                    )
            return fresh
        try:
            saved = json.loads(self._checkpoint_file.read_text(encoding="utf-8"))
            if not isinstance(saved, dict):
                raise ValueError("checkpoint não é um objeto JSON")
            if (saved.get("version") != self.CHECKPOINT_VERSION
                    or saved.get("fingerprint") != fingerprint
                    or type(saved.get("next_batch")) is not int
                    or not 0 <= saved["next_batch"] <= total_batches
                    or not isinstance(saved.get("agg"), dict)
                    or set(saved["agg"]) != set(empty_agg)
                    or any(not isinstance(saved["agg"][key], type(value))
                           for key, value in empty_agg.items())
                    or saved.get("max_risk") not in {"ALTO", "MEDIO", "BAIXO"}
                    or not isinstance(saved.get("rule_sources"), dict)
                    or not isinstance(saved.get("unverified_rules", []), list)
                    or not isinstance(saved.get("partial_batches", {}), dict)
                    or any(not isinstance(saved["rule_sources"].get(topic), list)
                           for topic in ("participacao", "selecao"))
                    or len(saved["rule_sources"]["participacao"]) != len(saved["agg"].get("requisitos_participacao", []))
                    or len(saved["rule_sources"]["selecao"]) != len(saved["agg"].get("selecao", []))
                    or not isinstance(saved.get("explanations"), dict)):
                raise ValueError("checkpoint incompatível")
            logger.info("Retomando análise: %s/%s lotes de extração já concluídos",
                        saved["next_batch"], total_batches)
            saved.setdefault("unverified_rules", [])
            saved.setdefault("partial_batches", {})
            if saved.get("explanation_version") != self.EXPLANATION_VERSION:
                if saved.get("explanation_version") == 3:
                    logger.info("Refazendo apenas explicações pendentes; mantendo as confirmadas e a extração")
                    saved["explanations"] = {
                        topic: {
                            key: row for key, row in
                            (saved["explanations"].get(topic) or {}).items()
                            if isinstance(row, dict)
                            and row.get("explicacao") != self.OLD_PENDING_EXPLANATION
                        }
                        for topic in ("participacao", "selecao")
                    }
                else:
                    logger.info("Refazendo explicações com as instruções atualizadas; mantendo a extração")
                    saved["explanations"] = {"participacao": {}, "selecao": {}}
                saved["explanation_version"] = self.EXPLANATION_VERSION
                self._save_checkpoint(saved)
            return saved
        except (OSError, ValueError, TypeError, KeyError) as exc:
            logger.warning("Checkpoint ilegível ou incompatível, reiniciando: %s", exc)
            return fresh

    def _save_checkpoint(self, state: dict) -> None:
        target = getattr(self, "_checkpoint_file", None)
        if target is None:
            return
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=target.parent,
                prefix="checkpoint_", suffix=".tmp", delete=False,
            ) as handle:
                temp_name = handle.name
                json.dump(state, handle, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, target)
        finally:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)

    @staticmethod
    def _explanation_context(batch: list[dict], agg: dict, chunks: list[dict]) -> dict:
        """Seleciona candidatos a evidência; somente citação literal será validada."""
        stop = {"edital", "item", "requisito", "licitacao", "participante",
                "empresa", "condicao", "condicoes", "proposta", "documento",
                "documentos", "apresentar", "verificar", "ser", "estar"}

        def terms(value: str) -> set[str]:
            normalized = unicodedata.normalize("NFKD", value.casefold())
            normalized = "".join(c for c in normalized if not unicodedata.combining(c))
            return {word[:6] for word in re.findall(r"[a-z0-9]+", normalized)
                    if len(word) >= 4 and word not in stop}

        other_items = agg.get("requisitos_participacao", []) + agg.get("selecao", [])
        context = {}
        for entry in batch:
            tokens = terms(entry["item"])
            related = sorted(
                (other for other in other_items if other != entry["item"]
                 and len(tokens & terms(other)) >= 2),
                key=lambda other: len(tokens & terms(other)), reverse=True,
            )[:3]
            scored = sorted(
                ((len(tokens & terms(chunk["content"])), chunk)
                 for chunk in chunks if chunk.get("content")),
                key=lambda pair: pair[0], reverse=True,
            )
            ranked = [chunk for score, chunk in scored if score >= 2][:2]
            # Uma palavra em comum sugere onde procurar, mas não valida citação.
            candidate_pages = list(dict.fromkeys(
                chunk.get("metadata", {}).get("page")
                for score, chunk in scored if score >= 1
                and chunk.get("metadata", {}).get("page") is not None
            ))[:2]
            context[str(entry["id"])] = {
                "itens_relacionados": related,
                "paginas_candidatas": candidate_pages,
                "trechos_candidatos": [
                    {"pagina": chunk.get("metadata", {}).get("page"),
                     "texto": chunk["content"][:1100]}
                    for chunk in ranked
                ],
            }
        return context

    @staticmethod
    def _valid_explanation_evidence(row: dict, candidates: list[dict]) -> dict | None:
        """Confere citação literal e página; similaridade sozinha não é evidência."""
        evidence = row.get("evidencia")
        if not isinstance(evidence, dict):
            return None
        quote = evidence.get("trecho")
        page = evidence.get("pagina")
        if not isinstance(quote, str) or not 15 <= len(quote.strip()) <= 240:
            return None
        normalized_quote = " ".join(quote.split()).casefold()
        for candidate in candidates:
            if (page == candidate["pagina"] and normalized_quote in
                    " ".join(candidate["texto"].split()).casefold()):
                return {"pagina": page, "trecho": quote.strip()}
        return None

    @staticmethod
    def _numeric_claims_supported(explanation: str, item: str, quote: str) -> bool:
        """Impede números novos na explicação sem apoio no item ou citação."""
        numbers = lambda value: set(re.findall(r"\d+(?:[.,]\d+)*", value))
        return numbers(explanation) <= numbers(item) | numbers(quote)

    @staticmethod
    def _source_context(source: dict, chunks: list[dict]) -> str:
        """Inclui o começo da cláusula anterior se ela estiver na mesma página."""
        for index, chunk in enumerate(chunks):
            if (chunk.get("metadata", {}).get("page") == source["pagina"]
                    and chunk.get("metadata", {}).get("chunk_id") == source["chunk_id"]):
                previous = chunks[index - 1] if index else None
                prefix = (previous["content"][-600:] if previous
                          and previous.get("metadata", {}).get("page") == source["pagina"]
                          else "")
                content = chunk["content"]
                offset = content.find(source["citacao"])
                start = max(0, offset - 550) if offset >= 0 else 0
                return (prefix + "\n" + content[start:start + 1450]).strip()
        return source["citacao"]

    @staticmethod
    def _mode_applicability(condition: str, cover: str) -> str | None:
        """Compara modalidades nomeadas na condição e no cabeçalho do documento."""
        selected = re.search(r"(?im)^\s*MODO\s+DE\s+DISPUTA\s*:\s*([^\n]+)", cover)
        alternative = re.search(
            r"(?i)modo\s+de\s+disputa\s*[\"“']([^\"”']+)[\"”']", condition
        )
        if not selected or not alternative:
            return None
        if not re.fullmatch(r"[\wÀ-ÿ ]{2,40}", selected.group(1).strip()):
            return None

        def normalize(value: str) -> str:
            value = unicodedata.normalize("NFKD", value.casefold())
            return " ".join("".join(c for c in value if not unicodedata.combining(c)).split())

        return ("aplicavel" if normalize(selected.group(1)) == normalize(alternative.group(1))
                else "nao_aplicavel")

    def _explain_all_requirements(
        self, agg: dict, checkpoint: dict | None = None,
        chunks: list[dict] | None = None,
        rule_sources: dict[str, list[dict]] | None = None,
    ) -> dict[str, list[dict]]:
        """Explica cada item extraído; recusa respostas com itens faltando ou trocados."""
        import json

        topics = {
            "participacao": agg["requisitos_participacao"],
            "selecao": agg["selecao"],
        }
        batch_size = 1 if self._output_budget() <= 1000 else 5
        tasks = [
            (topic, items, offset)
            for topic, items in topics.items()
            for offset in range(0, len(items), batch_size)
        ]
        explained: dict[str, list[dict]] = {topic: [] for topic in topics}
        saved = checkpoint.setdefault("explanations", {}) if checkpoint is not None else {}
        # Informação global do próprio documento, sem supor que haja capa ou modo.
        cover = "\n".join(chunk.get("content", "") for chunk in (chunks or [])
                          if chunk.get("metadata", {}).get("page") == 1)[:2500]

        for task_index, (topic, items, offset) in enumerate(tasks, 1):
            batch = [
                {"id": i + 1, "item": item, **({"fonte": {
                    **rule_sources[topic][i],
                    "contexto": self._source_context(rule_sources[topic][i], chunks or []),
                }}
                   if rule_sources is not None else {})}
                for i, item in enumerate(items[offset:offset + batch_size], offset)
            ]
            if _progress_callback:
                _progress_callback(
                    stage="Explicando todos os requisitos",
                    query_atual=f"{topic}: itens {offset + 1}–{offset + len(batch)}",
                    query_num=task_index,
                    query_total=len(tasks),
                    batch_atual=task_index,
                    batch_total=len(tasks),
                )
            stored_for_topic = saved.setdefault(topic, {}) if checkpoint is not None else {}
            explanations_by_id = {
                entry["id"]: stored_for_topic[str(entry["id"])]
                for entry in batch
                if isinstance(stored_for_topic.get(str(entry["id"])), dict)
                and stored_for_topic[str(entry["id"])].get("item") == entry["item"]
                and stored_for_topic[str(entry["id"])].get("explicacao")
            }
            # Acrescenta páginas de conferência a resultados já salvos, sem nova chamada à LLM.
            enriched = False
            for entry in batch:
                existing = explanations_by_id.get(entry["id"])
                if existing and existing.get("situacao") == "incerta" and "paginas_para_revisao" not in existing:
                    existing["paginas_para_revisao"] = (
                        [entry["fonte"]["pagina"]] if entry.get("fonte") else
                        self._explanation_context([entry], agg, chunks or [])[
                            str(entry["id"])]["paginas_candidatas"]
                    )
                    enriched = True
            if enriched and checkpoint is not None:
                self._save_checkpoint(checkpoint)
            invoked_this_task = False
            for attempt in range(3):
                missing = [entry for entry in batch if entry["id"] not in explanations_by_id]
                if not missing:
                    break
                if attempt:
                    logger.warning(
                        "Explicações faltantes em %s: IDs %s. Tentativa %s/3",
                        topic, [entry["id"] for entry in missing], attempt + 1,
                    )
                    if _progress_callback:
                        _progress_callback(
                            stage="Explicando todos os requisitos",
                            query_atual=f"{topic}: repetindo IDs {[entry['id'] for entry in missing]} ({attempt + 1}/3)",
                        )
                    time.sleep(8)
                prompt = (
                    "Explique TODOS os itens abaixo, um por um, em português simples para quem "
                    "nunca participou de uma licitação. Para cada ID, escreva até 2 frases "
                    "úteis: explique a regra e seu efeito prático, sem repetir o título. "
                    "Também forneça explicacao_basica: uma frase que esclareça apenas "
                    "o significado do título, sem dados, consequências, pessoas, etapas "
                    "ou condições que não estejam no próprio título. Essa frase será "
                    "mostrada como leitura preliminar se a evidência não for verificável. "
                    "Use o item e o contexto do mesmo documento. Se o item trouxer "
                    "fonte, ela foi copiada literalmente do PDF: explique APENAS essa "
                    "cláusula e sua condição; não substitua a página por outra busca. "
                    "Se a fonte contradisser o título, diga que há inconsistência "
                    "e use situacao=incerta. Somente quando não houver fonte: trechos "
                    "candidatos encontrados por palavras podem ser irrelevantes; "
                    "cite um trecho candidato ou declare situacao=incerta. "
                    "Classifique situacao como aplicavel, condicional, nao_aplicavel ou incerta. "
                    "Preserve todas as condições, exceções, etapas e sujeitos da regra. "
                    "Uma regra alternativa ou hipotética não se torna aplicável apenas "
                    "por estar descrita. Compare-a com os dados gerais deste documento "
                    "quando houver. Se houver conflito ou faltar contexto, use incerta. "
                    "Não invente dados, prazos, percentuais, obrigações, consequências ou "
                    "conclusões jurídicas. Não troque quem executa uma ação nem confunda "
                    "requisito, consequência e hipótese. Se houver regras aparentemente "
                    "incompatíveis, aponte a divergência sem decidir qual prevalece. "
                    "Regras gerais sobre cumprir o edital abrangem os requisitos concretos "
                    "explicados nos demais itens: não diga que o edital não lista suas condições. "
                    "Não acrescente 'o texto não detalha', 'consulte o edital' ou variações "
                    "por hábito; a interface já exibe um aviso geral. Mencione uma lacuna "
                    "somente se ela impedir a compreensão deste item e diga qual dado falta. "
                    "Não agrupe, omita ou renumere IDs. "
                    "Responda somente JSON no formato "
                    '{"explicacoes":[{"id":1,"explicacao":"Texto claro.",'
                    '"explicacao_basica":"Significado do título em linguagem simples.",'
                    '"situacao":"aplicavel","evidencia":{"pagina":1,'
                    '"trecho":"Frase literal do trecho candidato."}}]}. '
                    "Para situacao incerta, evidencia pode ser null. "
                    f"\nTEMA: {topic}\nCABEÇALHO DO DOCUMENTO: {cover}"
                    f"\nCONTEXTO POR ID: "
                    f"{json.dumps({} if rule_sources is not None else self._explanation_context(missing, agg, chunks or []), ensure_ascii=False)}"
                    f"\nITENS: {json.dumps(missing, ensure_ascii=False)}"
                )
                invoked_this_task = True
                response = self._invoke_with_rotation([
                    SystemMessage(content="Responda apenas JSON válido com todos os IDs recebidos."),
                    HumanMessage(content=prompt),
                ])
                parsed = self._parse_llm_json(response.content)
                rows = parsed.get("explicacoes") if isinstance(parsed, dict) else None
                requested_ids = {entry["id"] for entry in missing}
                if isinstance(rows, list):
                    for row in rows:
                        if (isinstance(row, dict) and type(row.get("id")) is int
                                and row["id"] in requested_ids
                                and isinstance(row.get("explicacao"), str)
                                and len(row["explicacao"].split()) >= 8):
                            entry = next(item for item in missing if item["id"] == row["id"])
                            source = entry.get("fonte")
                            context = ({"trechos_candidatos": [],
                                        "paginas_candidatas": [source["pagina"]]}
                                       if source else self._explanation_context(
                                           [entry], agg, chunks or [])[str(row["id"])])
                            candidates = context["trechos_candidatos"]
                            evidence = ({"pagina": source["pagina"], "trecho": source["citacao"]}
                                        if source else self._valid_explanation_evidence(row, candidates))
                            status = row.get("situacao")
                            if status not in {"aplicavel", "condicional", "nao_aplicavel", "incerta"}:
                                status = "incerta"
                            if source and source.get("condicao") and status == "aplicavel":
                                status = "condicional"
                            mode_status = None
                            if source and source.get("condicao"):
                                mode_status = self._mode_applicability(source["condicao"], cover)
                                if mode_status == "nao_aplicavel":
                                    status = "nao_aplicavel"
                            if status == "nao_aplicavel" and mode_status != "nao_aplicavel":
                                status = "condicional" if source and source.get("condicao") else "incerta"
                            if (source and not source.get("condicao") and status == "aplicavel"
                                    and re.search(r"(?i)\b(?:subitem supra|item anterior|procedimento de que trata)\b",
                                                  source["citacao"])):
                                status = "incerta"
                            basic = row.get("explicacao_basica")
                            if not (isinstance(basic, str) and len(basic.split()) >= 5
                                    and self._numeric_claims_supported(
                                        basic, entry["item"], ""
                                    )):
                                basic = None
                            if chunks is not None and (
                                evidence is None or status == "incerta"
                                or not self._numeric_claims_supported(
                                    row["explicacao"], entry["item"],
                                    evidence["trecho"] + " " + (source.get("condicao", "") if source else "")
                                )
                            ):
                                preliminary = (
                                    row["explicacao"].strip()
                                    if evidence and self._numeric_claims_supported(
                                        row["explicacao"], entry["item"],
                                        evidence["trecho"] + " " + (source.get("condicao", "") if source else "")
                                    ) else basic
                                ) or (
                                    row["explicacao"].strip()
                                    if self._numeric_claims_supported(
                                        row["explicacao"], entry["item"], ""
                                    ) else "O tópico menciona " + entry["item"].rstrip(".") + "."
                                )
                                result = {**entry, "situacao": "incerta", "evidencia": evidence,
                                          "condicao": source.get("condicao", "") if source else "",
                                          "explicacao": preliminary,
                                          "explicacao_preliminar": True,
                                          "paginas_para_revisao": context["paginas_candidatas"]}
                            else:
                                result = {**entry, "situacao": status, "evidencia": evidence,
                                          "condicao": source.get("condicao", "") if source else "",
                                          "explicacao": row["explicacao"].strip()}
                            explanations_by_id[row["id"]] = result
                            if checkpoint is not None:
                                stored_for_topic[str(row["id"])] = result
                    if checkpoint is not None:
                        self._save_checkpoint(checkpoint)
            missing_ids = [entry["id"] for entry in batch if entry["id"] not in explanations_by_id]
            if missing_ids:
                raise RuntimeError(
                    f"Explicação incompleta em {topic}, itens {offset + 1}–"
                    f"{offset + len(batch)} (IDs faltantes: {missing_ids}); relatório não publicado."
                )
            explained[topic].extend([explanations_by_id[entry["id"]] for entry in batch])
            if invoked_this_task and task_index < len(tasks):
                time.sleep(8)

        for topic, items in topics.items():
            if len(explained[topic]) != len(items):
                raise RuntimeError(f"Explicação incompleta em {topic}; relatório não publicado.")
        return explained

    def _process_with_rag_llm(self, chunks: list[dict]) -> dict:
        """
        Passagem unica: 1 chamada ao LLM por batch, extraindo TODAS as informacoes.
        31 batches em vez de 9 queries x 31 = 279 chamadas.
        Reducao de 90pct no consumo de tokens.
        """
        BATCH_SIZE    = 5
        SLEEP_BETWEEN = 8

        system_msg = SystemMessage(content=(
            "Voce e um especialista em licitacoes publicas brasileiras (Lei 14.133/2021). "
            "Responda sempre com JSON valido, sem markdown, sem texto fora do JSON."
        ))

        risk_order = {"ALTO": 2, "MEDIO": 1, "BAIXO": 0}

        # O TextChunker já limita os trechos. Preserve o conteúdo completo.
        all_batches = [chunks[i:i+BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)]
        total = len(all_batches)
        logger.info(f"Passagem unica: {len(chunks)} chunks -> {total} batches")

        # Acumuladores por campo
        agg = {
            "objeto": "",
            "documentos": [],
            "documentos_execucao": [],
            "anexos_referencia": [],
            "pendencias_documentais": [],
            "requisitos_participacao": [],
            "prazos": [],
            "custos": [],
            "selecao": [],
            "entregas": [],
            "eliminacao": [],
            "riscos": [],
            "nivel_risco": "BAIXO",
        }
        max_risk = "BAIXO"
        checkpoint = self._load_checkpoint(chunks, total, agg)
        agg = checkpoint["agg"]
        max_risk = checkpoint["max_risk"]

        def normalized(item):
            text = unicodedata.normalize("NFKD", item.casefold())
            text = "".join(c for c in text if not unicodedata.combining(c))
            return re.sub(r"[^\w]+", " ", text).strip()

        def add_unique(lst, items):
            seen = {normalized(existing) for existing in lst}
            for item in items:
                item = item.strip()
                key = normalized(item)
                if key and key not in seen:
                    lst.append(item)
                    seen.add(key)

        for b_idx in range(checkpoint["next_batch"], total):
            batch = all_batches[b_idx]
            if _progress_callback:
                total_itens = sum(len(v) for v in agg.values() if isinstance(v, list))
                _progress_callback(
                    stage="Analisando edital com IA",
                    query_atual="Processando trechos do edital",
                    query_num=b_idx + 1,
                    query_total=total,
                    batch_atual=b_idx + 1,
                    batch_total=total,
                    itens_encontrados=total_itens,
                )
            try:
                previous = []
                for older in reversed(chunks[:b_idx * BATCH_SIZE]):
                    if older.get("metadata", {}).get("page") != batch[0].get("metadata", {}).get("page"):
                        break
                    previous.insert(0, older)
                parsed, extracted_rules = self._extract_budgeted_batch(
                    batch, system_msg, previous, checkpoint, b_idx + 1
                )
                for candidate in parsed.get("_unverified_rules", []):
                    checkpoint["unverified_rules"].append({**candidate, "lote": b_idx + 1})

                if not agg["objeto"] and parsed.get("objeto"):
                    agg["objeto"] = parsed["objeto"]

                for field in ["documentos", "documentos_execucao", "anexos_referencia",
                               "pendencias_documentais", "prazos", "custos", "entregas",
                               "eliminacao", "riscos"]:
                    add_unique(agg[field], parsed.get(field, []))

                for topic, field in (("participacao", "requisitos_participacao"),
                                     ("selecao", "selecao")):
                    existing = {
                        (normalized(source["item"]), normalized(source["citacao"]),
                         normalized(source["condicao"]))
                        for source in checkpoint["rule_sources"][topic]
                    }
                    for source in extracted_rules[topic]:
                        key = (normalized(source["item"]), normalized(source["citacao"]),
                               normalized(source["condicao"]))
                        if key not in existing:
                            checkpoint["rule_sources"][topic].append(source)
                            agg[field].append(source["item"])
                            existing.add(key)

                batch_risk = parsed.get("nivel_risco", "BAIXO").upper()
                if risk_order.get(batch_risk, 0) > risk_order.get(max_risk, 0):
                    max_risk = batch_risk

                gained = sum(len(parsed.get(f, [])) for f in
                             ["documentos","prazos","eliminacao","riscos"]) + sum(
                                 len(rows) for rows in extracted_rules.values())
                if gained:
                    total_acc = sum(len(v) for v in agg.values() if isinstance(v, list))
                    logger.info(f"  batch {b_idx+1}/{total}: +{gained} itens (total={total_acc})")

                checkpoint.update(next_batch=b_idx + 1, agg=agg, max_risk=max_risk)
                checkpoint["partial_batches"].pop(str(b_idx + 1), None)
                self._save_checkpoint(checkpoint)

            except Exception as e:
                logger.error(f"  batch {b_idx+1} erro: {e}")
                raise RuntimeError(
                    f"Análise incompleta: falha no lote {b_idx + 1}/{total}; "
                    "os lotes anteriores e as partes concluídas foram salvos para retomada."
                ) from e

            if b_idx + 1 < total:
                time.sleep(SLEEP_BETWEEN)

        agg["nivel_risco"] = max_risk

        # O mesmo nome pode surgir em lotes diferentes com etapas conflitantes.
        # Deixe-o para conferência, em vez de publicá-lo como entrega do licitante.
        document_fields = ("documentos", "documentos_execucao", "anexos_referencia",
                           "pendencias_documentais")
        owners = {}
        originals = {}
        for field in document_fields:
            for item in agg[field]:
                key = normalized(item)
                owners.setdefault(key, set()).add(field)
                originals.setdefault(key, item)
        conflicts = {key for key, fields in owners.items() if len(fields) > 1}
        if conflicts:
            for field in document_fields:
                agg[field] = [item for item in agg[field] if normalized(item) not in conflicts]
            add_unique(agg["pendencias_documentais"],
                       [originals[key] for key in originals if key in conflicts])
        checkpoint["agg"] = agg
        self._save_checkpoint(checkpoint)

        # A recuperação temática consulta somente a coleção deste edital.
        # Os trechos e páginas acompanham o relatório para conferência.
        rag_sources = self._retrieve_rag_chunks(top_k=3)
        if not any(rag_sources.values()):
            raise RuntimeError("A busca vetorial não retornou trechos deste edital")
        source_refs = {
            topic: [
                {
                    "page": chunk.get("metadata", {}).get("page"),
                    "chunk_id": chunk.get("metadata", {}).get("chunk_id"),
                    "section": chunk.get("section", "geral"),
                    "excerpt": chunk["content"][:450],
                }
                for chunk in retrieved
            ]
            for topic, retrieved in rag_sources.items()
        }

        # Uma chamada adicional produz explicações após a extração. Falhas nesta
        # etapa ficam visíveis; os itens extraídos continuam disponíveis.
        explanatory_report = checkpoint.get("explanatory_report")
        explanatory_error = None
        if explanatory_report is None and any(
            agg[field] for field in ("requisitos_participacao", "selecao", "prazos", "riscos")
        ):
            try:
                if _progress_callback:
                    _progress_callback(stage="Redigindo resumo e requisitos")
                explanatory_report = self._synthesize_explanatory_report(agg, source_refs)
                checkpoint["explanatory_report"] = explanatory_report
                self._save_checkpoint(checkpoint)
            except Exception as exc:
                logger.exception("Falha ao redigir o relatório explicativo")
                explanatory_error = str(exc)

        # A síntese acima é uma amostra; esta etapa cobre TODOS os itens das duas
        # seções mostradas na interface, inclusive quando há muitos itens.
        detailed_explanations = self._explain_all_requirements(
            agg, checkpoint, chunks, checkpoint["rule_sources"]
        )

        # Mapeia para o formato compativel com display e Node 4
        # rag_answers simula as respostas por pergunta para a tab "Perguntas Respondidas"
        rag_answers = {
            "objeto_licitacao": {
                "label": "Objeto da Licitacao",
                "resposta": agg["objeto"],
                "detalhes": agg["entregas"],
                "nivel_risco": "BAIXO",
                "observacao": "",
            },
            "posso_participar": {
                "label": "Posso participar?",
                "resposta": f"{len(agg['requisitos_participacao'])} requisitos identificados." if agg["requisitos_participacao"] else "Sem restricoes especificas identificadas.",
                "detalhes": agg["requisitos_participacao"],
                "nivel_risco": "BAIXO",
                "observacao": "",
            },
            "prazos": {
                "label": "Quais sao os prazos?",
                "resposta": f"{len(agg['prazos'])} prazos identificados." if agg["prazos"] else "Nenhum prazo especifico identificado.",
                "detalhes": agg["prazos"],
                "nivel_risco": "BAIXO",
                "observacao": "",
            },
            "custos_pagamento": {
                "label": "Quanto custa e como pago?",
                "resposta": f"{len(agg['custos'])} informacoes de custo/pagamento." if agg["custos"] else "Sem informacoes de custo identificadas.",
                "detalhes": agg["custos"],
                "nivel_risco": "BAIXO",
                "observacao": "",
            },
            "selecao_processo": {
                "label": "Como sera a selecao ou avaliacao?",
                "resposta": f"{len(agg['selecao'])} criterios identificados." if agg["selecao"] else "Criterios nao identificados.",
                "detalhes": agg["selecao"],
                "nivel_risco": "BAIXO",
                "observacao": "",
            },
            "objeto_escopo": {
                "label": "O que devo entregar ou produzir?",
                "resposta": f"{len(agg['entregas'])} entregaveis identificados." if agg["entregas"] else "Entregaveis nao identificados.",
                "detalhes": agg["entregas"],
                "nivel_risco": "BAIXO",
                "observacao": "",
            },
            "eliminacao_penalidades": {
                "label": "Quais sao as regras de eliminacao?",
                "resposta": f"{len(agg['eliminacao'])} regras identificadas." if agg["eliminacao"] else "Sem regras de eliminacao identificadas.",
                "detalhes": agg["eliminacao"],
                "nivel_risco": max_risk,
                "observacao": "",
            },
            "documentos": {
                "label": "Documentos para proposta ou habilitação",
                "resposta": f"{len(agg['documentos'])} documentos candidatos; confirme as exigências no edital." if agg["documentos"] else "Nenhum documento de proposta ou habilitação identificado.",
                "detalhes": agg["documentos"],
                "nivel_risco": "BAIXO",
                "observacao": "",
            },
            "pontos_criticos": {
                "label": "Pontos criticos e riscos",
                "resposta": f"{len(agg['riscos'])} riscos identificados. Nivel: {max_risk}." if agg["riscos"] else "Nenhum risco critico identificado.",
                "detalhes": agg["riscos"],
                "nivel_risco": max_risk,
                "observacao": "",
            },
        }

        llm_analysis = {
            "objeto": agg["objeto"],
            "requisitos_tecnicos": agg["requisitos_participacao"],
            "documentos_exigidos": agg["documentos"],
            "prazos": agg["prazos"],
            "pontos_criticos": list(dict.fromkeys(agg["eliminacao"] + agg["riscos"])),
            "nivel_risco": max_risk,
            "recomendacoes": agg["requisitos_participacao"][:5],
        }

        structured_analysis = {
            0: {
                "technical_requirements": agg["requisitos_participacao"],
                "documentation": llm_analysis["documentos_exigidos"],
                "deadlines": agg["prazos"],
                "object_info": {"description": agg["objeto"], "quantity": None, "unit": None},
                "risk_factors": llm_analysis["pontos_criticos"],
                "section": "full_scan",
            }
        }
        critical_analysis = {
            0: {
                "critical_points": llm_analysis["pontos_criticos"],
                "risk_level": max_risk,
                "recommendations": agg["requisitos_participacao"][:5],
                "ambiguities": [],
                "section": "full_scan",
            }
        }

        return {
            "total_chunks_processed": len(chunks),
            "structured_analysis": structured_analysis,
            "critical_analysis": critical_analysis,
            "llm_analysis": llm_analysis,
            "rag_answers": rag_answers,
            "selection_process": agg["selecao"],
            "execution_items": agg["entregas"],
            "documentos_execucao": agg["documentos_execucao"],
            "anexos_referencia": agg["anexos_referencia"],
            "pendencias_documentais": agg["pendencias_documentais"],
            "detailed_explanations": detailed_explanations,
            "rule_sources": checkpoint["rule_sources"],
            "unverified_rules": checkpoint["unverified_rules"],
            "rag_sources": source_refs,
            "explanatory_report": explanatory_report,
            "explanatory_error": explanatory_error,
        }

    def process_from_node2(self, chunks: list[dict]) -> dict:
        """Processa todos os chunks usando LLM real ou análise heurística explícita."""
        logger.info(f"Processando {len(chunks)} chunks do Nó 2")

        if not chunks:
            return {
                "total_chunks_processed": 0,
                "structured_analysis": {},
                "critical_analysis": {},
            }

        if not self.mock_mode:
            return self._process_with_rag_llm(chunks)

        # Modo de demonstração: mantém a estrutura esperada pelo workflow
        # e pelo checklist, sem chamar a API externa.
        structured_analysis = {
            chunk.get("metadata", {}).get("chunk_id", i): self.extract_structured_info(chunk)
            for i, chunk in enumerate(chunks)
        }
        critical_analysis = {
            chunk.get("metadata", {}).get("chunk_id", i): self.identify_critical_points(chunk)
            for i, chunk in enumerate(chunks)
        }
        return {
            "total_chunks_processed": len(chunks),
            "structured_analysis": structured_analysis,
            "critical_analysis": critical_analysis,
        }

    def process_complete_analysis(self, node2_output: dict) -> dict:
        """
        Processa análise completa usando saída do Nó 2.

        Args:
            node2_output: Saída do Nó 2 com embeddings

        Returns:
            Dicionário com análise completa integrada
        """
        logger.info("Iniciando análise completa integrada")

        chunks = node2_output.get("chunks_with_embeddings", [])

        # Processa análise
        analysis = self.process_from_node2(chunks)

        # Combina análises
        combined = self.combine_analyses(chunks)

        # Gera resumo executivo
        summary = self.generate_executive_summary(combined)

        return {
            "summary": summary,
            "technical_analysis": analysis["structured_analysis"],
            "risk_assessment": analysis["critical_analysis"],
            "recommendations": combined["priority_recommendations"],
            "total_chunks_processed": analysis["total_chunks_processed"],
        }

    def combine_analyses(self, chunks: list[dict]) -> dict:
        """
        Combina análise estruturada e crítica.

        Args:
            chunks: Lista de chunks

        Returns:
            Dicionário com análises combinadas
        """
        logger.info("Combinando análises estruturada e crítica")

        all_structured = []
        all_critical = []
        all_risk_levels = []

        for chunk in chunks:
            structured = self.extract_structured_info(chunk)
            critical = self.identify_critical_points(chunk)

            all_structured.append(structured)
            all_critical.append(critical)
            all_risk_levels.append(critical["risk_level"])

        # Determina nível de risco combinado
        combined_risk = self._calculate_overall_risk(all_risk_levels)

        # Recomendações prioritárias
        priority_recommendations = []
        for critical in all_critical:
            priority_recommendations.extend(critical["recommendations"])

        return {
            "structured_info": all_structured,
            "critical_info": all_critical,
            "combined_risk_level": combined_risk,
            "priority_recommendations": list(
                set(priority_recommendations)
            ),  # Remove duplicatas
        }

    def generate_executive_summary(self, analysis_result: dict) -> dict:
        """
        Gera resumo executivo da análise.

        Args:
            analysis_result: Resultado da análise combinada

        Returns:
            Dicionário com resumo executivo
        """
        logger.info("Gerando resumo executivo")

        structured = analysis_result.get("structured_info", [])

        # Extrai principais descobertas
        key_findings = []
        for struct in structured:
            if struct.get("technical_requirements"):
                key_findings.extend(struct["technical_requirements"][:2])
            if struct.get("deadlines"):
                key_findings.extend(struct["deadlines"][:1])

        # Resumo de risco
        risk_summary = f"Nível de risco geral: {analysis_result.get('combined_risk_level', 'BAIXO')}"

        # Próximos passos
        next_steps = analysis_result.get("priority_recommendations", [])

        return {
            "overview": f"Análise de {len(structured)} chunks processados",
            "key_findings": key_findings[:5],  # Limita a 5 descobertas principais
            "risk_summary": risk_summary,
            "next_steps": next_steps[:3],  # Limita a 3 próximos passos
        }

    def validate_integration_output(self, output: dict) -> bool:
        """
        Valida a saída da integração.

        Args:
            output: Dicionário de saída da integração

        Returns:
            True se válido, False caso contrário
        """
        required_fields = [
            "total_chunks_processed",
            "structured_analysis",
            "critical_analysis",
        ]

        for field in required_fields:
            if field not in output:
                logger.error(f"Campo obrigatório ausente: {field}")
                return False

        # Verifica se total_chunks_processed é inteiro
        if not isinstance(output["total_chunks_processed"], int):
            logger.error("total_chunks_processed deve ser um inteiro")
            return False

        logger.info("Validação de integração concluída")
        return True
