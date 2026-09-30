import io
import json
import os
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from app.rag import gemini_client as module


def response(text='{"topics":[]}', finish='STOP'):
    return {'candidates': [{'finishReason': finish, 'content': {'parts': [{'text': text}]}}],
            'usageMetadata': {'promptTokenCount': 100, 'candidatesTokenCount': 30}}


def error(code, details=None, headers=None):
    body = json.dumps({'error': {'details': details or []}}).encode()
    return HTTPError('https://example.invalid', code, 'error', headers or {}, io.BytesIO(body))


class GeminiTests(unittest.TestCase):
    def setUp(self):
        module._states.clear()
        env = patch.dict(os.environ, {'GEMINI_API_KEY': 'test-secret', 'GEMINI_MODEL': 'gemini-3.5-flash-lite',
            'GEMINI_RPM_BUDGET': '10', 'GEMINI_TPM_BUDGET': '100000', 'GEMINI_RPD_BUDGET': '500',
            'GEMINI_OUTPUT_TOKEN_BUDGET': '8192', 'TOPIC_LLM_PROVIDER': 'gemini'}, clear=True)
        env.start(); self.addCleanup(env.stop)
        self.clock, self.waits, self.events = [100.0], [], []
        def sleep(delay): self.waits.append(delay); self.clock[0] += delay
        for name, replacement in [('monotonic', lambda: self.clock[0]), ('time', lambda: self.clock[0]), ('sleep', sleep)]:
            p = patch.object(time, name, replacement); p.start(); self.addCleanup(p.stop)

    def client(self, transport=None):
        return module.GeminiTopicClient(progress=lambda **kw: self.events.append(kw),
                                       transport=transport or (lambda body: response()))

    def test_rest_configuration_and_json_schema(self):
        bodies = []
        client = self.client(lambda body: bodies.append(body) or response())
        self.assertEqual(client.invoke('system', {'passages': []}), {'topics': []})
        config = bodies[0]['generationConfig']
        self.assertEqual(config['maxOutputTokens'], 8192)
        self.assertEqual(config['responseSchema']['required'], ['topics'])
        self.assertEqual(config['thinkingConfig'], {'thinkingLevel': 'minimal'})
        self.assertEqual(module.GeminiTopicClient._schema({'topics': []})['required'], ['themes'])
        self.assertEqual(client.state['minute'][0][1], 100)

    def test_rpm_shared_between_jobs_and_visible_wait(self):
        a, b = self.client(), self.client()
        a.invoke('system', {'passages': []}); b.invoke('system', {'passages': []})
        self.assertAlmostEqual(sum(self.waits), 6.1)
        self.assertTrue(any(e['activity'] == 'waiting' and e['wait_until'] for e in self.events))

    def test_tpm_waits_until_old_input_expires(self):
        with patch.dict(os.environ, {'GEMINI_TPM_BUDGET': '150'}):
            client = self.client()
            client._reserve(100); client._reserve(100)
        self.assertAlmostEqual(sum(self.waits), 60)

    def test_daily_local_limit_blocks_without_extra_call(self):
        with patch.dict(os.environ, {'GEMINI_RPD_BUDGET': '1'}):
            client = self.client()
            client.invoke('system', {})
            with self.assertRaisesRegex(RuntimeError, 'diária local'):
                client.invoke('system', {})

    def test_oversized_input_is_not_sent(self):
        with patch.dict(os.environ, {'GEMINI_TPM_BUDGET': '1'}):
            with self.assertRaises(module.GeminiRequestTooLarge): self.client().invoke('system', {})

    def test_429_retries_with_delay_and_progress(self):
        calls = []
        def transport(body):
            calls.append(body)
            if len(calls) == 1: raise error(429, headers={'Retry-After': '12'})
            return response()
        self.client(transport).invoke('system', {})
        self.assertEqual(len(calls), 2)
        self.assertAlmostEqual(sum(self.waits), 13)
        self.assertTrue(any(e['activity'] == 'retrying' for e in self.events))

    def test_persistent_transient_errors_stop(self):
        calls = []
        def transport(body): calls.append(body); raise error(503)
        with self.assertRaisesRegex(RuntimeError, 'três novas tentativas'): self.client(transport).invoke('system', {})
        self.assertEqual(len(calls), 4)

    def test_auth_error_has_no_secret_or_retry(self):
        with self.assertRaisesRegex(RuntimeError, 'credencial') as exc:
            self.client(lambda body: (_ for _ in ()).throw(error(403))).invoke('system', {})
        self.assertNotIn('test-secret', str(exc.exception)); self.assertEqual(self.waits, [])

    def test_daily_server_quota_stops_and_blocks_further_jobs(self):
        client = self.client(lambda body: (_ for _ in ()).throw(error(429, details=[{
            '@type': 'QuotaFailure', 'violations': [{'quotaId': 'GenerateRequestsPerDayPerProject'}]}])))
        with self.assertRaisesRegex(RuntimeError, 'Cota Gemini indisponível'): client.invoke('system', {})
        with self.assertRaisesRegex(RuntimeError, 'cinco minutos'): client.invoke('system', {})
        self.assertEqual(self.waits, [])

    def test_truncation_and_invalid_json_are_not_published(self):
        with self.assertRaises(module.GeminiOutputTruncated):
            self.client(lambda body: response('{', 'MAX_TOKENS')).invoke('system', {})
        with self.assertRaisesRegex(ValueError, 'JSON inválido'):
            self.client(lambda body: response('{')).invoke('system', {})

    def test_key_sent_as_header_not_url(self):
        captured = []
        class FakeResponse:
            def __enter__(self): return io.StringIO(json.dumps(response()))
            def __exit__(self, *args): pass
        def open_request(request, timeout): captured.append(request); return FakeResponse()
        with patch.object(module, 'urlopen', open_request):
            module.GeminiTopicClient().invoke('system', {})
        self.assertNotIn('test-secret', captured[0].full_url)
        self.assertEqual(captured[0].get_header('X-goog-api-key'), 'test-secret')

    def test_selection_and_missing_key(self):
        self.assertEqual(module.validate_topic_configuration(), 'gemini')
        with patch.dict(os.environ, {'GEMINI_API_KEY': ''}):
            with self.assertRaisesRegex(ValueError, 'GEMINI_API_KEY'): module.validate_topic_configuration()


if __name__ == '__main__': unittest.main()
