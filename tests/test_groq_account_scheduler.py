"""Scheduler tests without network calls or optional LangChain dependencies."""
import ast
import logging
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import os
import pytest


@pytest.fixture
def client(monkeypatch):
    source = Path(__file__).parents[1] / 'app/rag/node_3_analyzer.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    namespace = dict(ChatGroq=object, Path=Path, time=time, os=os, deque=deque, logger=logging.getLogger(__name__), re=__import__('re'))
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(source), 'exec'), namespace)
    node = object.__new__(namespace['Node3RequirementAnalyzer'])
    node._api_keys = ['account-a', 'account-b', 'account-c']
    node._current_key_idx = 0
    node.max_tokens = 2500
    node._output_token_cap = 950
    node._rpm_budget_override = 1
    node.llm = 'account-a'
    node._make_llm = lambda key: key
    clock, waits = [100.0], []
    monkeypatch.setenv('GROQ_INDEPENDENT_ACCOUNTS', 'true')
    monkeypatch.setattr(time, 'monotonic', lambda: clock[0])
    def sleep(seconds):
        waits.append(seconds)
        clock[0] += seconds
    monkeypatch.setattr(time, 'sleep', sleep)
    return node, clock, waits


def test_other_accounts_run_before_waiting(client):
    node, clock, waits = client
    for expected in [0, 1, 2, 2]:
        node._select_available_account(['short'])
        assert node._current_key_idx == expected
        node._before_groq_request(['short'])
    assert waits == [60]


def test_keep_current_account_if_it_has_capacity(client):
    node, _, waits = client
    node._select_available_account(['short'])
    node._select_available_account(['short'])
    assert node._current_key_idx == 0
    assert waits == []


def test_quota_is_isolated_and_reset_reopens_account(client):
    node, clock, waits = client
    node._record_quota({'x-ratelimit-remaining-requests': '0', 'x-ratelimit-reset-requests': '2s'})
    node._select_available_account(['short'])
    assert node._current_key_idx == 1
    assert 'requests_remaining' not in node._quota
    clock[0] += 3
    node._activate_account(0)
    node._select_available_account(['short'])
    assert node._current_key_idx == 0
    assert waits == []


def test_429_switches_without_sleep_and_recovers(client):
    node, _, waits = client
    class Limited(Exception):
        status_code = 429
        response = SimpleNamespace(headers={'retry-after': '40'})
        body = {'error': {'type': 'requests', 'message': 'Rate limit'}}
    called = []
    def invoke(messages):
        called.append(node._current_key_idx)
        if len(called) == 1:
            raise Limited('429')
        return 'ok'
    node._invoke_groq = invoke
    assert node._invoke_with_rotation(['short']) == 'ok'
    assert called == [0, 1]
    assert waits == []


def test_all_daily_exhausted_terminates(client):
    node, _, waits = client
    class Daily(Exception):
        status_code = 429
        body = {'error': {'message': 'tokens per day exhausted'}}
    node._invoke_groq = lambda messages: (_ for _ in ()).throw(Daily('429 TPD'))
    with pytest.raises(RuntimeError, match='nenhuma conta'):
        node._invoke_with_rotation(['short'])
    assert waits == []


def test_repeated_429_has_finite_attempts(client):
    node, _, waits = client
    class Limited(Exception):
        status_code = 429
        body = {'error': {'type': 'requests'}}
    called = []
    def invoke(messages):
        called.append(node._current_key_idx)
        raise Limited('429')
    node._invoke_groq = invoke
    with pytest.raises(RuntimeError, match='três esperas'):
        node._invoke_with_rotation(['short'])
    assert len(called) <= 12
    assert all(w <= 60 for w in waits)


def test_same_organization_mode_is_default(client, monkeypatch):
    node, _, _ = client
    monkeypatch.delenv('GROQ_INDEPENDENT_ACCOUNTS')
    assert not node._independent_accounts()


def test_duplicate_keys_are_removed(client, monkeypatch):
    node, _, _ = client
    monkeypatch.setenv('GROQ_API_KEY', 'duplicate')
    monkeypatch.setenv('GROQ_API_KEY_2', 'duplicate')
    monkeypatch.setenv('GROQ_API_KEY_3', 'other')
    monkeypatch.delenv('GROQ_API_KEY_4', raising=False)
    monkeypatch.delenv('GROQ_API_KEY_5', raising=False)
    assert node._load_api_keys() == ['duplicate', 'other']
