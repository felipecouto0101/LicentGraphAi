import os
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class StartupTests(unittest.TestCase):
    def test_real_api_and_status_work_with_heavy_imports_blocked(self):
        code = '''
import importlib.abc, sys
blocked = {'langgraph', 'torch', 'transformers', 'sentence_transformers', 'chromadb',
           'pdfplumber', 'langchain_groq', 'langchain_google_genai'}
class BlockHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked or fullname == 'app.rag.langgraph_workflow':
            raise AssertionError('Unexpected startup dependency: ' + fullname)
sys.meta_path.insert(0, BlockHeavy())
from app.api.main import app
from fastapi.testclient import TestClient
with TestClient(app) as client:
    for route in ['/', '/status']:
        assert client.get(route).status_code == 200
    assert client.get('/status').json()['ready'] is True
assert not any(n.split('.')[0] in blocked for n in sys.modules)
print('Lightweight API startup passed')
'''
        env = {**os.environ, 'TOPIC_LLM_PROVIDER': 'gemini', 'GEMINI_API_KEY': 'fake',
               'NODE3_MOCK_MODE': 'false'}
        result = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).parents[1],
                                env=env, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Lightweight API startup passed', result.stdout)

    def test_config_preserves_explicit_demo_and_missing_key_validation(self):
        from app.analysis_config import is_mock_mode, validate_analysis_configuration
        with patch.dict(os.environ, {'NODE3_MOCK_MODE': 'false', 'GROQ_API_KEY': ''}):
            self.assertFalse(is_mock_mode())
            with self.assertRaisesRegex(ValueError, 'GROQ_API_KEY'):
                validate_analysis_configuration()
        with patch.dict(os.environ, {'NODE3_MOCK_MODE': 'true', 'GROQ_API_KEY': ''}):
            self.assertTrue(is_mock_mode())
            validate_analysis_configuration()

    def test_full_job_loads_pipeline_on_demand_and_cleans_callback(self):
        from app.api import main as api
        workflow = types.ModuleType('app.rag.langgraph_workflow')
        calls = []
        def run(path, profile):
            calls.append((path, profile))
            return {'chunks': [], 'analysis': {}, 'checklist': {}}
        workflow.run_licit_graph_pipeline = run
        analyzer = types.ModuleType('app.rag.node_3_analyzer')
        api._jobs['startup-test'] = {'progress': {}}
        try:
            with patch.dict(sys.modules, {'app.rag.langgraph_workflow': workflow,
                                         'app.rag.node_3_analyzer': analyzer}):
                api._run_pipeline_job('startup-test', 'edital.pdf', None)
            self.assertEqual(calls, [('edital.pdf', None)])
            self.assertEqual(api._jobs['startup-test']['status'], 'done')
            self.assertIsNone(analyzer._progress_callback)
        finally:
            api._jobs.pop('startup-test', None)

    def test_pipeline_import_failure_becomes_job_error_without_cleanup_import(self):
        import builtins
        from app.api import main as api
        original = builtins.__import__
        imports = []
        def guarded(name, *args, **kwargs):
            imports.append(name)
            if name == 'app.rag.langgraph_workflow':
                raise ImportError('pipeline dependency missing')
            return original(name, *args, **kwargs)
        api._jobs['startup-test'] = {'progress': {}}
        try:
            with patch('builtins.__import__', side_effect=guarded):
                api._run_pipeline_job('startup-test', 'edital.pdf', None)
            self.assertEqual(api._jobs['startup-test']['status'], 'error')
            self.assertIn('pipeline dependency missing', api._jobs['startup-test']['error'])
            self.assertNotIn('app.rag.node_3_analyzer', imports)
        finally:
            api._jobs.pop('startup-test', None)


if __name__ == '__main__': unittest.main()
