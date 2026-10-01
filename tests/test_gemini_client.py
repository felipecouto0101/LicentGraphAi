import os
import time
import unittest
from unittest.mock import patch
import httpx
from google.genai.errors import ClientError, ServerError
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from app.rag import gemini_client as module


def response(text='{"topics":[]}', finish='STOP'):
    return AIMessage(content=text, response_metadata={'finish_reason': finish},
                     usage_metadata={'input_tokens': 100, 'output_tokens': 30, 'total_tokens': 130})


def error(code, details=None, headers=None):
    cls = ServerError if code >= 500 else ClientError
    return cls(code, {'error': {'message': 'test', 'details': details or []}},
               response=httpx.Response(code, headers=headers or {}))


class FakeLLM:
    def __init__(self, callback): self.callback = callback
    def invoke(self, messages, **kwargs):
        return self.callback({'messages': messages, **kwargs})


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
                                       llm=FakeLLM(transport or (lambda body: response())))

    def test_langchain_messages_and_json_schema(self):
        bodies = []
        client = self.client(lambda body: bodies.append(body) or response())
        self.assertEqual(client.invoke('system', {'passages': []}), {'topics': []})
        config = bodies[0]
        self.assertEqual(config['response_mime_type'], 'application/json')
        self.assertEqual(config['response_json_schema']['required'], ['topics'])
        self.assertIsInstance(config['messages'][0], SystemMessage)
        self.assertIsInstance(config['messages'][1], HumanMessage)
        self.assertIn(module.protect_messages([SystemMessage(content='system')])[0].content, config['messages'][0].content)
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
            client.invoke('system', {'passages': []})
            with self.assertRaisesRegex(RuntimeError, 'diária local'):
                client.invoke('system', {'passages': []})

    def test_oversized_input_is_not_sent(self):
        with patch.dict(os.environ, {'GEMINI_TPM_BUDGET': '1'}):
            with self.assertRaises(module.GeminiRequestTooLarge): self.client().invoke('system', {'passages': []})

    def test_429_retries_with_delay_and_progress(self):
        calls = []
        def transport(body):
            calls.append(body)
            if len(calls) == 1: raise error(429, headers={'Retry-After': '12'})
            return response()
        self.client(transport).invoke('system', {'passages': []})
        self.assertEqual(len(calls), 2)
        self.assertAlmostEqual(sum(self.waits), 13)
        self.assertTrue(any(e['activity'] == 'retrying' for e in self.events))

    def test_persistent_transient_errors_stop(self):
        calls = []
        def transport(body): calls.append(body); raise error(503)
        with self.assertRaisesRegex(RuntimeError, 'três novas tentativas'): self.client(transport).invoke('system', {'passages': []})
        self.assertEqual(len(calls), 4)

    def test_auth_error_has_no_secret_or_retry(self):
        with self.assertRaisesRegex(RuntimeError, 'credencial') as exc:
            self.client(lambda body: (_ for _ in ()).throw(error(403))).invoke('system', {'passages': []})
        self.assertNotIn('test-secret', str(exc.exception)); self.assertEqual(self.waits, [])

    def test_daily_server_quota_stops_and_blocks_further_jobs(self):
        client = self.client(lambda body: (_ for _ in ()).throw(error(429, details=[{
            '@type': 'QuotaFailure', 'violations': [{'quotaId': 'GenerateRequestsPerDayPerProject'}]}])))
        with self.assertRaisesRegex(RuntimeError, 'Cota Gemini indisponível'): client.invoke('system', {'passages': []})
        with self.assertRaisesRegex(RuntimeError, 'cinco minutos'): client.invoke('system', {'passages': []})
        self.assertEqual(self.waits, [])

    def test_truncation_and_invalid_json_are_not_published(self):
        with self.assertRaises(module.GeminiOutputTruncated):
            self.client(lambda body: response('{', 'MAX_TOKENS')).invoke('system', {'passages': []})
        with self.assertRaisesRegex(ValueError, 'JSON inválido'):
            self.client(lambda body: response('{')).invoke('system', {'passages': []})

    def test_factory_uses_explicit_key_developer_api_and_no_internal_retries(self):
        with patch.dict(os.environ, {'GOOGLE_API_KEY': 'other-key', 'GOOGLE_GENAI_USE_VERTEXAI': 'true'}):
            with patch('langchain_google_genai.ChatGoogleGenerativeAI') as factory:
                factory.return_value.invoke.return_value = response()
                client = module.GeminiTopicClient()
                client.invoke('system', {'passages': []})
                client.invoke('system', {'passages': []})
        self.assertEqual(factory.call_count, 1)
        options = factory.call_args.kwargs
        self.assertEqual(options['api_key'], 'test-secret')
        self.assertFalse(options['vertexai'])
        self.assertEqual(options['max_retries'], 0)
        self.assertEqual(options['max_tokens'], 8192)
        self.assertEqual(options['thinking_level'], 'minimal')
        self.assertEqual(options['timeout'], 120)

    def test_real_langchain_adapter_without_network(self):
        from langchain_google_genai import ChatGoogleGenerativeAI
        from google.genai import types
        llm = ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite', api_key='fake',
                                    vertexai=False, max_retries=0, max_tokens=8192,
                                    timeout=120, thinking_level='minimal', temperature=0.2)
        result = types.GenerateContentResponse(candidates=[types.Candidate(
            content=types.Content(role='model', parts=[types.Part(text='{"topics":[]}')]),
            finish_reason='STOP')], usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=100, candidates_token_count=30, total_token_count=130))
        with patch.object(llm.client.models, 'generate_content', return_value=result) as generate:
            client = module.GeminiTopicClient(llm=llm)
            self.assertEqual(client.invoke('system', {'passages': []}), {'topics': []})
        config = generate.call_args.kwargs['config']
        self.assertEqual(config.response_json_schema['required'], ['topics'])
        self.assertEqual(config.max_output_tokens, 8192)
        self.assertEqual(config.http_options.retry_options.attempts, 0)
        self.assertEqual(client.state['minute'][0][1], 100)

    def test_wrapped_sdk_error_preserves_retry_details(self):
        calls = []
        def transport(body):
            calls.append(body)
            if len(calls) == 1:
                try:
                    raise error(429, details=[{'retryDelay': '15s'}])
                except ClientError as cause:
                    raise RuntimeError('SDK wrapped error') from cause
            return response()
        self.client(transport).invoke('system', {'passages': []})
        self.assertEqual(len(calls), 2)
        self.assertAlmostEqual(sum(self.waits), 16)

    def test_real_sdk_does_not_add_hidden_retries(self):
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite', api_key='fake',
                                    vertexai=False, max_retries=0, timeout=120)
        # Exercise the real SDK retry layer, intercepting only the HTTP attempt.
        with patch.object(llm.client._api_client, '_request_once', side_effect=error(503)) as request:
            with self.assertRaisesRegex(RuntimeError, 'três novas tentativas'):
                module.GeminiTopicClient(llm=llm, progress=lambda **kw: self.events.append(kw)).invoke('system', {'passages': []})
        self.assertEqual(request.call_count, 4)
        self.assertEqual([e['retry'] for e in self.events if e['activity'] == 'retrying'], [1, 2, 3])

    def test_content_blocks_exclude_thoughts(self):
        msg = response()
        msg.content = [{'type': 'reasoning', 'reasoning': 'private'},
                       {'type': 'text', 'text': '{"topics":[]}' }]
        self.assertEqual(self.client(lambda body: msg).invoke('system', {'passages': []}), {'topics': []})

    def test_unknown_local_errors_are_not_retried(self):
        with self.assertRaisesRegex(ValueError, 'configuration'):
            self.client(lambda body: (_ for _ in ()).throw(ValueError('configuration'))).invoke('system', {'passages': []})
        self.assertEqual(self.waits, [])

    def test_network_timeout_is_retried(self):
        calls = []
        def transport(body):
            calls.append(body)
            if len(calls) == 1: raise httpx.ReadTimeout('timeout')
            return response()
        self.client(transport).invoke('system', {'passages': []})
        self.assertEqual(len(calls), 2)

    def test_selection_and_missing_key(self):
        self.assertEqual(module.validate_topic_configuration(), 'gemini')
        with patch.dict(os.environ, {'GEMINI_API_KEY': ''}):
            with self.assertRaisesRegex(ValueError, 'GEMINI_API_KEY'): module.validate_topic_configuration()

    def test_missing_key_never_selects_another_provider(self):
        with patch.dict(os.environ, {'GEMINI_API_KEY': '', 'TOPIC_LLM_PROVIDER': ''}):
            self.assertEqual(module.provider_name(), 'gemini')
            with self.assertRaisesRegex(ValueError, 'GEMINI_API_KEY'):
                module.validate_topic_configuration()

    def test_unsupported_provider_fails_with_migration_instruction(self):
        with patch.dict(os.environ, {'TOPIC_LLM_PROVIDER': 'unsupported'}):
            with self.assertRaisesRegex(ValueError, 'Somente Gemini'):
                module.validate_topic_configuration()

    def test_full_analysis_shares_quota_with_topic_map_and_preserves_messages(self):
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer
        calls = []
        node = Node3RequirementAnalyzer(mock_mode=False)
        node.llm.llm = FakeLLM(lambda body: calls.append(body) or response('Resposta explicativa.'))
        topics = self.client()
        topics.invoke('system', {'passages': []})
        messages = [SystemMessage(content='Explain'), HumanMessage(content='Source')]
        result = node._invoke_llm(messages)
        self.assertEqual(result.content, 'Resposta explicativa.')
        self.assertEqual(calls[0]['messages'], module.protect_messages(messages))
        self.assertNotIn('response_json_schema', calls[0])
        self.assertIs(node.llm.state, topics.state)
        self.assertAlmostEqual(sum(self.waits), 6.1)

    def test_full_analysis_uses_real_langchain_adapter_with_explicit_model_and_key(self):
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer
        with patch('langchain_google_genai.ChatGoogleGenerativeAI') as factory:
            factory.return_value.invoke.return_value = response('{"documentos":[]}')
            node = Node3RequirementAnalyzer(api_key='explicit-key', model_name='gemini-2.5-flash', max_tokens=1500)
            result = node._invoke_llm([HumanMessage(content='Source')])
        self.assertEqual(result.content, '{"documentos":[]}')
        self.assertEqual(factory.call_args.kwargs['api_key'], 'explicit-key')
        self.assertEqual(factory.call_args.kwargs['model'], 'gemini-2.5-flash')
        self.assertEqual(factory.call_args.kwargs['max_tokens'], 1500)
        self.assertEqual(factory.call_args.kwargs['max_retries'], 0)

    def test_truncation_blocks_full_analysis_before_returning_partial_content(self):
        from app.rag.node_3_analyzer import Node3RequirementAnalyzer
        node = Node3RequirementAnalyzer(mock_mode=False)
        node.llm.llm = FakeLLM(lambda body: response('{', 'MAX_TOKENS'))
        with self.assertRaises(module.GeminiOutputTruncated):
            node._invoke_llm([HumanMessage(content='Source')])


if __name__ == '__main__': unittest.main()
