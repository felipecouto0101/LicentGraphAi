"""Checks polling messages without requiring an installed Streamlit runtime."""
import ast
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace


class FrontendTests(unittest.TestCase):
    def test_wait_and_countdown_render_alongside_last_progress(self):
        calls = []
        st = SimpleNamespace(container=lambda **kw: nullcontext(), fragment=lambda **kw: lambda fn: fn)
        for name in ('subheader', 'write', 'progress', 'caption', 'warning', 'info', 'rerun'):
            setattr(st, name, lambda *args, _name=name, **kw: calls.append((_name, args, kw)))
        job = {'status': 'running', 'progress': {'stage': 'Identificando subtemas', 'completed': 2, 'total': 10,
            'activity': 'waiting', 'activity_message': 'Aguardando cota Gemini', 'wait_until': 130,
            'provider': 'gemini', 'model': 'gemini-3.5-flash-lite'}}
        requests = SimpleNamespace(get=lambda *a, **kw: SimpleNamespace(status_code=200, json=lambda: job),
                                   RequestException=RuntimeError)
        tree = ast.parse((Path(__file__).parents[1] / 'app/streamlit_app.py').read_text(encoding='utf-8'))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_poll')
        namespace = {'st': st, 'requests': requests, 'time': SimpleNamespace(time=lambda: 100, sleep=lambda _: None)}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), 'frontend', 'exec'), namespace)
        namespace['_poll']('http://localhost', 'job')
        self.assertTrue(any(c[0] == 'warning' and 'Aguardando cota' in c[1][0] for c in calls))
        self.assertTrue(any(c[0] == 'caption' and '30 segundos' in c[1][0] for c in calls))
        self.assertTrue(any(c[0] == 'progress' and c[1] == (0.2,) for c in calls))


if __name__ == '__main__': unittest.main()
