"""Gemini REST client for the topic map, with shared local quota reservations."""
import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import deque
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

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
    def __init__(self, progress=None, transport=None):
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
        self.transport = transport or self._post
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
            time.sleep(min(wait, 30))
        self._event('requesting', 'Gemini está identificando e organizando os assuntos.',
                    local_requests_remaining=remaining, provider='gemini', model=self.model)
        return reservation

    def _post(self, body):
        request = Request(
            f'https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent',
            data=json.dumps(body, ensure_ascii=False).encode(),
            headers={'Content-Type': 'application/json', 'x-goog-api-key': self.key}, method='POST')
        with urlopen(request, timeout=120) as response:
            return json.load(response)

    @staticmethod
    def _retry_delay(error):
        headers = getattr(error, 'headers', None) or {}
        try:
            value = headers.get('Retry-After', headers.get('retry-after'))
            delay = float(value) if value is not None else None
        except (TypeError, ValueError):
            delay = None
        body = {}
        if isinstance(error, HTTPError):
            try:
                body = json.loads(error.read().decode())
            except (ValueError, OSError):
                pass
        details = body.get('error', {}).get('details', [])
        for item in details:
            if isinstance(item, dict) and 'retryDelay' in item:
                try:
                    delay = max(delay or 0, float(str(item['retryDelay']).rstrip('s')))
                except ValueError:
                    pass
        daily = any(('perday' in json.dumps(item).lower().replace('_', '')
                     or 'per_day' in json.dumps(item).lower()) for item in details)
        return max(1, (delay if delay is not None else 86400 if daily else 60) + 1), daily

    @staticmethod
    def _schema(payload):
        string = {'type': 'STRING'}
        if 'passages' in payload:
            source = {'type': 'OBJECT', 'properties': {
                'id': string, 'lines': {'type': 'ARRAY', 'items': {'type': 'INTEGER'}}},
                'required': ['id', 'lines']}
            topic = {'type': 'OBJECT', 'properties': {'theme': string, 'title': string,
                     'sources': {'type': 'ARRAY', 'items': source}},
                     'required': ['theme', 'title', 'sources']}
            return {'type': 'OBJECT', 'properties': {'topics': {'type': 'ARRAY', 'items': topic}},
                    'required': ['topics']}
        child = {'type': 'OBJECT', 'properties': {'title': string,
                 'source_ids': {'type': 'ARRAY', 'items': string}}, 'required': ['title', 'source_ids']}
        theme = {'type': 'OBJECT', 'properties': {'title': string,
                 'subtopics': {'type': 'ARRAY', 'items': child}}, 'required': ['title', 'subtopics']}
        return {'type': 'OBJECT', 'properties': {'themes': {'type': 'ARRAY', 'items': theme}},
                'required': ['themes']}

    def invoke(self, system, payload):
        text = json.dumps(payload, ensure_ascii=False)
        # Estimativa conservadora em bytes; o consumo real vem em usageMetadata.
        estimated = len((system + text).encode('utf-8')) + 64
        body = {'systemInstruction': {'parts': [{'text': system}]},
                'contents': [{'role': 'user', 'parts': [{'text': text}]}],
                'generationConfig': {'temperature': 0.2, 'maxOutputTokens': self.output,
                                     'responseMimeType': 'application/json',
                                     'responseSchema': self._schema(payload)}}
        if self.model == 'gemini-3.5-flash-lite':
            body['generationConfig']['thinkingConfig'] = {'thinkingLevel': 'minimal'}
        for attempt in range(4):
            reservation = self._reserve(estimated)
            try:
                result = self.transport(body)
            except (HTTPError, URLError, TimeoutError) as exc:
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
            usage = result.get('usageMetadata', {})
            actual = usage.get('promptTokenCount')
            if isinstance(actual, int) and actual > 0:
                with _lock:
                    reservation[1] = actual
            self._event('validating', 'Conferindo os assuntos e suas referências ao PDF.',
                        input_tokens=actual, output_tokens=usage.get('candidatesTokenCount'))
            candidates = result.get('candidates') or []
            if not candidates:
                raise RuntimeError('Gemini não retornou conteúdo para este lote.')
            candidate = candidates[0]
            if candidate.get('finishReason') == 'MAX_TOKENS':
                raise GeminiOutputTruncated('Resposta Gemini truncada; dividindo lote antes de publicar.')
            if candidate.get('finishReason') not in (None, 'STOP'):
                raise RuntimeError('Gemini interrompeu a resposta; lote não publicado.')
            content = ''.join(part.get('text', '') for part in candidate.get('content', {}).get('parts', [])
                              if not part.get('thought'))
            try:
                parsed = json.loads(content)
            except ValueError:
                raise ValueError('Gemini retornou JSON inválido; corrigir o formato.') from None
            if not isinstance(parsed, dict):
                raise ValueError('Gemini deve retornar um objeto JSON.')
            logger.info('Gemini: modelo=%s, entrada=%s, saída=%s', self.model, actual, usage.get('candidatesTokenCount'))
            return parsed
