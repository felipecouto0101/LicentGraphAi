import unittest
from concurrent.futures import Future
from pathlib import Path
from threading import Event
from unittest.mock import patch
from types import SimpleNamespace
from app.ui_health import APIHealth, check_api


class Executor:
    def __init__(self): self.calls = []
    def submit(self, fn, url):
        future = Future()
        self.calls.append((fn, url, future))
        return future


class HealthTests(unittest.TestCase):
    def test_pending_check_never_waits_or_submits_duplicate(self):
        health, executor = APIHealth(), Executor()
        self.assertEqual(health.poll('url', executor, now=0)['state'], 'checking')
        self.assertEqual(health.poll('url', executor, now=10)['state'], 'checking')
        self.assertEqual(len(executor.calls), 1)

    def test_snapshot_reused_until_ttl_then_refreshed(self):
        health, executor = APIHealth(), Executor()
        health.poll('url', executor, now=0)
        executor.calls[0][2].set_result({'state': 'online', 'status': {'ready': True}})
        self.assertEqual(health.poll('url', executor, now=1)['state'], 'online')
        health.poll('url', executor, now=15)
        self.assertEqual(len(executor.calls), 1)
        health.poll('url', executor, now=16)
        self.assertEqual(len(executor.calls), 2)

    def test_changed_url_does_not_publish_old_result(self):
        health, executor = APIHealth(), Executor()
        health.poll('old', executor, now=0)
        executor.calls[0][2].set_result({'state': 'online', 'status': {}})
        self.assertEqual(health.poll('new', executor, now=1)['state'], 'checking')
        self.assertEqual(executor.calls[-1][1], 'new')

    def test_failure_is_cached_and_can_recover(self):
        health, executor = APIHealth(ttl=1), Executor()
        health.poll('url', executor, now=0)
        executor.calls[0][2].set_exception(RuntimeError('failure'))
        self.assertEqual(health.poll('url', executor, now=1)['state'], 'unavailable')
        health.poll('url', executor, now=2)
        executor.calls[-1][2].set_result({'state': 'online', 'status': {}})
        self.assertEqual(health.poll('url', executor, now=3)['state'], 'online')

    def test_health_request_has_short_timeout_and_handles_invalid_response(self):
        response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: [])
        with patch('app.ui_health.requests.get', return_value=response) as get:
            self.assertEqual(check_api('url')['state'], 'unavailable')
        self.assertEqual(get.call_args.kwargs['timeout'], (1, 1))


class StreamlitRuntimeTests(unittest.TestCase):
    def test_initial_screen_finishes_while_status_request_is_blocked(self):
        from streamlit.testing.v1 import AppTest
        release = Event()
        def slow_status(*args, **kw):
            release.wait(10)
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'ready': True})
        app = AppTest.from_file(str(Path(__file__).parents[1] / 'app/streamlit_app.py'))
        try:
            with patch('app.ui_health.requests.get', side_effect=slow_status):
                app.run(timeout=4)
                self.assertFalse(app.exception)
                self.assertTrue(any(s.value == 'Comece pelo seu edital' for s in app.subheader))
                self.assertFalse(release.is_set())
                self.assertEqual(app.session_state['api_health'].snapshot['state'], 'checking')
        finally:
            release.set()

    def test_closed_source_does_not_render_document_content(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(Path(__file__).parents[1] / 'app/streamlit_app.py'))
        app.session_state['job_id'] = 'job'
        app.session_state['result'] = {'map_version': 4, 'organization_status': 'done', 'page_count': 1,
            'topic_map': [{'title': 'Participação', 'subtopics': [{'title': 'Inscrição', 'pages': [1],
                'evidence': [{'quote': 'SOURCE_MARKER', 'page': 1, 'line': 1, 'source_id': 'c1'}],
                'source_blocks': [{'text': 'SOURCE_MARKER', 'page': 1}]}]}]}
        with patch('app.ui_health.check_api', return_value={'state': 'online', 'status': {'ready': True}}):
            app.run(timeout=4)
        self.assertFalse(app.exception)
        self.assertTrue(any(s.value == 'Inscrição' for s in app.subheader))
        self.assertFalse(any('SOURCE_MARKER' in s.value for s in app.markdown))
        app.session_state['sources:job:0:0'] = True
        with patch('app.ui_health.check_api', return_value={'state': 'online', 'status': {'ready': True}}):
            app.run(timeout=4)
        self.assertFalse(app.exception)
        self.assertTrue(any('SOURCE_MARKER' in s.value for s in app.markdown))

    def test_repeated_theme_titles_do_not_collide(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(Path(__file__).parents[1] / 'app/streamlit_app.py'))
        app.session_state['job_id'] = 'job'
        app.session_state['result'] = {'map_version': 4, 'organization_status': 'done',
            'topic_map': [{'title': 'Regras', 'subtopics': [{'title': 'Assunto', 'pages': [1], 'evidence': []}]},
                          {'title': 'Regras', 'subtopics': [{'title': 'Outro assunto', 'pages': [2], 'evidence': []}]}]}
        with patch('app.ui_health.check_api', return_value={'state': 'online', 'status': {'ready': True}}):
            app.run(timeout=4)
        self.assertFalse(app.exception)
        self.assertFalse(any(s.value == 'Outro assunto' for s in app.subheader))


class LauncherTests(unittest.TestCase):
    def test_both_processes_start_before_waiting_and_stop_on_interrupt(self):
        import start_app
        from unittest.mock import Mock
        trace = []
        process = Mock()
        process.poll.side_effect = lambda: trace.append('poll') or None
        with patch.object(start_app, 'start_fastapi', side_effect=lambda: trace.append('api') or process), \
             patch.object(start_app, 'start_streamlit', side_effect=lambda: trace.append('ui') or process), \
             patch.object(start_app.time, 'sleep', side_effect=KeyboardInterrupt):
            self.assertEqual(start_app.main(), 0)
        self.assertEqual(trace[:2], ['api', 'ui'])
        self.assertEqual(process.terminate.call_count, 2)

    def test_launch_failure_cleans_up_started_backend(self):
        import start_app
        from unittest.mock import Mock
        process = Mock()
        process.poll.return_value = None
        with patch.object(start_app, 'start_fastapi', return_value=process), \
             patch.object(start_app, 'start_streamlit', side_effect=OSError('failure')):
            self.assertEqual(start_app.main(), 1)
        process.terminate.assert_called_once()


if __name__ == '__main__': unittest.main()
