"""Gemini LangChain client for the topic map, with shared local quota reservations."""
import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import deque

from app import telemetry

logger = logging.getLogger(__name__)
_states = {}
_lock = threading.Lock()


class GeminiRequestTooLarge(RuntimeError):
    pass


class GeminiOutputTruncated(GeminiRequestTooLarge):
    pass


def provider_name():
    provider = os.getenv('TOPIC_LLM_PROVIDER', '').strip().lower()
    if not provider:
        provider = 'gemini' if os.getenv('GEMINI_API_KEY', '').strip() else 'groq'
    if provider not in ('gemini', 'groq'):
        raise ValueError('TOPIC_LLM_PROVIDER deve ser gemini ou groq.')
    return provider


def validate_topic_configuration():
    provider = provider_name()
    name = 'GEMINI_API_KEY' if provider == 'gemini' else 'GROQ_API_KEY'
    if not os.getenv(name, '').strip() or os.getenv(name) == 'your_groq_api_key_here':
        raise ValueError(f'Configure {name} para organizar os assuntos.')
    return provider


class GeminiTopicClient:
    def __init__(self, progress=None, llm=None):
        self.key = os.getenv('GEMINI_API_KEY', '').strip()
        if not self.key:
            raise ValueError('Configure GEMINI_API_KEY no .env.')
        self.model = os.getenv('GEMINI_MODEL', 'gemini-3.5-flash-lite')
        if not re.fullmatch(r'[a-zA-Z0-9_.-]+', self.model):
            raise ValueError('GEMINI_MODEL inválido.')
        self.rpm = max(1, int(os.getenv('GEMINI_RPM_BUDGET', '10')))
        self.tpm = max(1, int(os.getenv('GEMINI_TPM_BUDGET', '100000')))
        self.rpd = max(1, int(os.getenv('GEMINI_RPD_BUDGET', '500')))
        self.output = max(1, int(os.getenv('GEMINI_OUTPUT_TOKEN_BUDGET', '8192')))
        self.progress = progress or (lambda **kw: None)
        self.llm = llm
        # Mesmo projeto com mais de uma chave deve compartilhar esse identificador.
        group = os.getenv('GEMINI_QUOTA_GROUP') or hashlib.sha256(self.key.encode()).hexdigest()
        self.state_key = (group, self.model)
        with _lock:
            self.state = _states.setdefault(self.state_key, {'minute': deque(), 'day': deque(), 'cooldown': 0})

    def _event(self, activity, message, **kw):
        self.progress(activity=activity, activity_message=message, wait_until=None, **kw)

    def _reserve(self, estimated):
        if estimated > self.tpm:
            raise GeminiRequestTooLarge('Entrada excede o orçamento local de tokens/min; reduza o lote.')
        while True:
            now = time.monotonic()
            with _lock:
                minute, day = self.state['minute'], self.state['day']
                while minute and now - minute[0][0] >= 60:
                    minute.popleft()
                while day and now - day[0] >= 86400:
                    day.popleft()
                if len(day) >= self.rpd:
                    raise RuntimeError('Cota diária local do Gemini atingida; subtemas validados preservados na sessão.')
                ready = self.state['cooldown']
                if minute:
                    # Espaçamento preventivo, sem confundir RPM com o tempo de geração.
                    ready = max(ready, minute[-1][0] + 60 / self.rpm + 0.1)
                used = sum(item[1] for item in minute)
                if used + estimated > self.tpm:
                    for stamp, tokens in minute:
                        used -= tokens
                        if used + estimated <= self.tpm:
                            ready = max(ready, stamp + 60.1)
                            break
                if ready <= now:
                    reservation = [now, estimated]
                    minute.append(reservation)
                    day.append(now)
                    remaining = self.rpd - len(day)
                    break
                wait = ready - now
            if wait > 300:
                raise RuntimeError('Gemini exige uma espera superior a cinco minutos; retome depois. '
                                   'Subtemas validados preservados na sessão.')
            self.progress(activity='waiting', activity_message='Aguardando a próxima janela de cota do Gemini.',
                          wait_until=time.time() + wait)
            seconds = min(wait, 30)
            with telemetry.stage("llm.wait", provider="gemini"):
                time.sleep(seconds)
            telemetry.quota_wait("gemini", seconds)
        self._event('requesting', 'Gemini está identificando e organizando os assuntos.',
                    local_requests_remaining=remaining, provider='gemini', model=self.model)
        return reservation

    def _invoke_model(self, system, text, schema):
        from langchain_core.messages import SystemMessage, HumanMessage
        if self.llm is None:
            from langchain_google_genai import ChatGoogleGenerativeAI
            options = dict(model=self.model, api_key=self.key, vertexai=False,
                           temperature=0.2, max_tokens=self.output, timeout=120,
                           max_retries=0)
            if self.model == 'gemini-3.5-flash-lite':
                options['thinking_level'] = 'minimal'
            self.llm = ChatGoogleGenerativeAI(**options)
        # Keep the raw AIMessage: inspect finish_reason and usage before parsing.
        return self.llm.invoke([SystemMessage(content=system), HumanMessage(content=text)],
                               response_mime_type='application/json', response_json_schema=schema)

    @staticmethod
    def _api_error(error):
        """LangChain wraps SDK errors; preserve status, headers and quota details."""
        from google.genai.errors import APIError
        import httpx
        seen = set()
        while error is not None and id(error) not in seen:
            seen.add(id(error))
            if isinstance(error, (APIError, httpx.TransportError, TimeoutError, ConnectionError)):
                return error
            error = error.__cause__ or error.__context__
        return None

    @staticmethod
    def _retry_delay(error):
        response = getattr(error, 'response', None)
        headers = getattr(response, 'headers', None) or {}
        try:
            value = headers.get('Retry-After', headers.get('retry-after'))
            delay = float(value) if value is not None else None
        except (TypeError, ValueError):
            delay = None
        body = getattr(error, 'details', {})
        body = body if isinstance(body, dict) else {}
        details = body.get('error', body).get('details', [])
        for item in details:
            if isinstance(item, dict) and 'retryDelay' in item:
                try:
                    delay = max(delay or 0, float(str(item['retryDelay']).rstrip('s')))
                except ValueError:
                    pass
        daily = any('perday' in json.dumps(item).lower().replace('_', '') for item in details)
        return max(1, (delay if delay is not None else 86400 if daily else 60) + 1), daily

    @staticmethod
    def _schema(payload):
        string = {'type': 'string'}
        if 'passages' in payload:
            source = {'type': 'object', 'properties': {
                'id': string, 'lines': {'type': 'array', 'items': {'type': 'integer'}}},
                'required': ['id', 'lines']}
            topic = {'type': 'object', 'properties': {'theme': string, 'title': string,
                     'sources': {'type': 'array', 'items': source}},
                     'required': ['theme', 'title', 'sources']}
            return {'type': 'object', 'properties': {'topics': {'type': 'array', 'items': topic}},
                    'required': ['topics']}
        child = {'type': 'object', 'properties': {'title': string,
                 'source_ids': {'type': 'array', 'items': string}}, 'required': ['title', 'source_ids']}
        theme = {'type': 'object', 'properties': {'title': string,
                 'subtopics': {'type': 'array', 'items': child}}, 'required': ['title', 'subtopics']}
        return {'type': 'object', 'properties': {'themes': {'type': 'array', 'items': theme}},
                'required': ['themes']}

    def invoke(self, system, payload):
        text = json.dumps(payload, ensure_ascii=False)
        # Estimativa conservadora em bytes; o consumo real vem em usageMetadata.
        estimated = len((system + text).encode('utf-8')) + 64
        schema = self._schema(payload)
        for attempt in range(4):
            reservation = self._reserve(estimated)
            try:
                with telemetry.stage("llm.request", provider="gemini", model=self.model):
                    result = self._invoke_model(system, text, schema)
            except Exception as error:
                exc = self._api_error(error)
                if exc is None:
                    raise
                status = getattr(exc, 'code', None)
                if status in (401, 403):
                    raise RuntimeError('Gemini recusou a credencial ou permissão do projeto.') from None
                if status is not None and status not in (429, 500, 502, 503, 504):
                    raise RuntimeError(f'Gemini recusou a chamada (HTTP {status}); confira modelo e configuração.') from None
                delay, daily = self._retry_delay(exc) if status == 429 else ((5, 15, 30, 30)[attempt], False)
                if daily or delay > 300:
                    with _lock:
                        self.state['cooldown'] = max(self.state['cooldown'], time.monotonic() + delay)
                    raise RuntimeError('Cota Gemini indisponível; retome após sua renovação. '
                                       'Subtemas validados preservados na sessão.') from None
                if attempt == 3:
                    raise RuntimeError('Gemini continuou indisponível após três novas tentativas; '
                                       'subtemas validados preservados na sessão.') from None
                with _lock:
                    self.state['cooldown'] = max(self.state['cooldown'], time.monotonic() + delay)
                self._event('retrying', f'Gemini indisponível; nova tentativa {attempt + 1}/3.', retry=attempt + 1)
                continue
            usage = result.usage_metadata or {}
            telemetry.llm_usage("gemini", usage.get("input_tokens"), usage.get("output_tokens"))
            actual = usage.get('input_tokens')
            if isinstance(actual, int) and actual > 0:
                with _lock:
                    reservation[1] = actual
            self._event('validating', 'Conferindo os assuntos e suas referências ao PDF.',
                        input_tokens=actual, output_tokens=usage.get('output_tokens'))
            finish = str(result.response_metadata.get('finish_reason', 'STOP')).split('.')[-1]
            if finish == 'MAX_TOKENS':
                raise GeminiOutputTruncated('Resposta Gemini truncada; dividindo lote antes de publicar.')
            if finish != 'STOP':
                raise RuntimeError('Gemini interrompeu a resposta; lote não publicado.')
            content = result.content
            if isinstance(content, list):
                content = ''.join(part.get('text', '') for part in content
                                  if isinstance(part, dict) and part.get('type') == 'text'
                                  and not part.get('thought'))
            if not content:
                raise RuntimeError('Gemini não retornou conteúdo para este lote.')
            try:
                parsed = json.loads(content)
            except ValueError:
                raise ValueError('Gemini retornou JSON inválido; corrigir o formato.') from None
            if not isinstance(parsed, dict):
                raise ValueError('Gemini deve retornar um objeto JSON.')
            logger.info('Gemini: modelo=%s, entrada=%s, saída=%s', self.model, actual, usage.get('output_tokens'))
            return parsed

