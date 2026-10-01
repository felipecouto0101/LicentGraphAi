"""Real SDK checks for correlation, privacy, handled failures and OTLP export."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.sampling import ALWAYS_ON
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import SimpleLogRecordProcessor, InMemoryLogRecordExporter
from app import telemetry


@pytest.fixture
def capture(monkeypatch):
    spans, logs, metrics = InMemorySpanExporter(), InMemoryLogRecordExporter(), InMemoryMetricReader()
    resource = Resource.create({'service.name': 'licitgraphai-api'})
    tracer = TracerProvider(resource=resource, sampler=ALWAYS_ON)
    tracer.add_span_processor(SimpleSpanProcessor(spans))
    meter = MeterProvider(resource=resource, metric_readers=[metrics])
    logger = LoggerProvider(resource=resource)
    logger.add_log_record_processor(SimpleLogRecordProcessor(logs))
    runtime = telemetry.Runtime(tracer, meter, logger)
    monkeypatch.setattr(telemetry, '_runtime', runtime)
    yield runtime, spans, logs, metrics
    runtime.shutdown()


def points(reader):
    return [(m.name, p) for r in reader.get_metrics_data().resource_metrics
            for s in r.scope_metrics for m in s.metrics for p in m.data.data_points]


def test_http_context_survives_thread_and_labels_use_route_templates(capture):
    _, spans, logs, metrics = capture
    app = FastAPI()
    app.add_middleware(telemetry.TelemetryMiddleware)
    @telemetry.observed_job('test')
    def job(job_id):
        with telemetry.stage('pdf.read'):
            telemetry.llm_usage('gemini', 10, 3)
    @app.get('/job/{job_id}')
    def route(job_id):
        telemetry.start_thread(job, (job_id,)).join(timeout=2)
        return {'ok': True}
    with TestClient(app) as client:
        assert client.get('/job/random-private-id?key=secret').status_code == 200
    finished = spans.get_finished_spans()
    assert len({s.context.trace_id for s in finished}) == 1
    child = next(s for s in finished if s.name == 'pdf.read')
    assert child.attributes['job.id'] == 'random-private-id'
    assert any('random-private-id' in r.log_record.body for r in logs.get_finished_logs())
    for _, point in points(metrics):
        assert 'random-private-id' not in json.dumps(dict(point.attributes))
    http = next(s for s in finished if s.name.startswith('GET'))
    assert http.attributes['http.route'] == '/job/{job_id}'
    assert 'secret' not in str(http.attributes)


def test_exception_details_and_document_content_are_not_exported(capture):
    _, spans, logs, metrics = capture
    with pytest.raises(ValueError):
        with telemetry.stage('map.organize'):
            raise ValueError('private PDF text and API key')
    span = spans.get_finished_spans()[0]
    assert span.status.status_code.name == 'ERROR'
    assert span.attributes['error.type'] == 'ValueError'
    assert not span.events
    assert 'private PDF' not in str(logs.get_finished_logs())
    assert any(p.attributes.get('outcome') == 'error' for _, p in points(metrics))


def test_caught_job_failure_is_not_reported_as_success(capture):
    _, spans, _, metrics = capture
    namespace = {'telemetry': telemetry, '_jobs': {
        'job': {'status': 'done', 'result': {'organization_status': 'partial'}}}}
    exec('@telemetry.observed_job("organization")\ndef worker(job_id):\n    return None', namespace)
    namespace['worker']('job')
    assert spans.get_finished_spans()[0].status.status_code.name == 'ERROR'
    data = points(metrics)
    assert any(n == 'licit_operations' and p.attributes['outcome'] == 'error' for n, p in data)
    assert next(p.value for n, p in data if n == 'licit_active_jobs') == 0
    assert telemetry._job_id.get() is None


def test_wait_and_usage_use_numeric_counts_only(capture):
    _, _, _, metrics = capture
    telemetry.quota_wait('groq', 6)
    telemetry.llm_usage('groq', 120, 50)
    data = points(metrics)
    assert next(p.sum for n, p in data if n == 'licit_quota_wait') == 6
    assert sum(p.value for n, p in data if n == 'licit_llm_tokens') == 170


def test_disabled_configuration_creates_no_sdk_runtime(monkeypatch):
    monkeypatch.setattr(telemetry, '_runtime', None)
    monkeypatch.setenv('OBSERVABILITY_ENABLED', 'false')
    telemetry.configure()
    with telemetry.stage('disabled') as span:
        assert span is None
    assert telemetry._runtime is None


def test_real_otlp_exporters_send_all_three_signals(monkeypatch):
    received = []
    class Collector(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append((self.path, self.rfile.read(int(self.headers['Content-Length']))))
            self.send_response(200)
            self.end_headers()
        def log_message(self, *args):
            pass
    server = HTTPServer(('127.0.0.1', 0), Collector)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv('OBSERVABILITY_ENABLED', 'true')
    monkeypatch.setenv('OTEL_TRACES_SAMPLER_ARG', '1.0')
    monkeypatch.setenv('OTEL_EXPORTER_OTLP_ENDPOINT', f'http://127.0.0.1:{server.server_port}')
    monkeypatch.setattr(telemetry, '_runtime', None)
    try:
        telemetry.configure()
        with telemetry.stage('smoke'):
            telemetry.llm_usage('gemini', 10, 5)
        for provider in telemetry._runtime.providers:
            assert provider.force_flush(timeout_millis=3000)
        assert {path for path, _ in received} == {'/v1/traces', '/v1/metrics', '/v1/logs'}
        assert all(body for _, body in received)
    finally:
        telemetry.shutdown()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_handled_failures_are_counted_when_trace_sampling_is_zero(capture):
    from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
    runtime, _, _, metrics = capture
    unsampled = TracerProvider(sampler=ALWAYS_OFF)
    runtime.tracer = unsampled.get_tracer('unsampled')
    try:
        with telemetry.stage('sampled.out') as span:
            assert not span.is_recording()
            telemetry.mark_failed(span)
        assert any(n == 'licit_operations' and p.attributes['outcome'] == 'error'
                   for n, p in points(metrics))
    finally:
        unsampled.shutdown()
