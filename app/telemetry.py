"""Optional bounded OTLP telemetry; never export documents or credentials."""
import contextvars
import functools
import json
import logging
import os
import threading
import time
from contextlib import contextmanager

_job_id = contextvars.ContextVar('telemetry_job_id', default=None)
_outcome = contextvars.ContextVar('telemetry_outcome', default=None)
_runtime = None
_events = logging.getLogger('app.telemetry.events')
_events.propagate = False


class EventFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps(record.event_fields, ensure_ascii=False)


class Runtime:
    def __init__(self, tracer, meter, logs):
        self.providers = (logs, meter, tracer)
        self.otel_logger = logs.get_logger("licitgraphai")
        self.tracer = tracer.get_tracer('licitgraphai')
        instruments = meter.get_meter('licitgraphai')
        self.duration = instruments.create_histogram('licit_stage_duration', unit='s')
        self.completed = instruments.create_counter('licit_operations', unit='1')
        self.active = instruments.create_up_down_counter('licit_active_jobs', unit='1')
        self.tokens = instruments.create_counter('licit_llm_tokens', unit='1')
        self.wait = instruments.create_histogram('licit_quota_wait', unit='s')
        self.http_duration = instruments.create_histogram('licit_http_duration', unit='s')

    def shutdown(self):
        for provider in self.providers:
            provider.shutdown()


def configure():
    """Load SDK/exporter threads only when enabled; no collector handshake."""
    global _runtime
    if _runtime is not None or os.getenv('OBSERVABILITY_ENABLED', 'false').lower() != 'true':
        return
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
    endpoint = os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://127.0.0.1:4318').rstrip('/')
    ratio = float(os.getenv('OTEL_TRACES_SAMPLER_ARG', '1.0'))
    if not 0 <= ratio <= 1:
        raise ValueError('OTEL_TRACES_SAMPLER_ARG must be between 0 and 1')
    resource = Resource.create({'service.name': 'licitgraphai-api',
                                'deployment.environment.name': os.getenv('APP_ENV', 'local')})
    tracer = TracerProvider(resource=resource, sampler=ParentBased(TraceIdRatioBased(ratio)))
    tracer.add_span_processor(BatchSpanProcessor(
        OTLPSpanExporter(endpoint=endpoint + '/v1/traces', timeout=2),
        max_queue_size=2048, schedule_delay_millis=5000))
    meter = MeterProvider(resource=resource, metric_readers=[PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=endpoint + '/v1/metrics', timeout=2),
        export_interval_millis=10000, export_timeout_millis=3000)])
    logs = LoggerProvider(resource=resource)
    logs.add_log_record_processor(BatchLogRecordProcessor(
        OTLPLogExporter(endpoint=endpoint + '/v1/logs', timeout=2),
        max_queue_size=2048, schedule_delay_millis=5000))
    _events.setLevel(logging.INFO)
    if not any(isinstance(h, logging.StreamHandler) for h in _events.handlers):
        stream = logging.StreamHandler()
        stream.setFormatter(EventFormatter())
        _events.addHandler(stream)
    _runtime = Runtime(tracer, meter, logs)
    event('telemetry.started')


def shutdown():
    global _runtime
    if _runtime is not None:
        runtime, _runtime = _runtime, None
        runtime.shutdown()


def event(name, **attributes):
    runtime = _runtime
    if runtime is None:
        return
    from opentelemetry import trace
    context = trace.get_current_span().get_span_context()
    fields = {'event': name, **attributes}
    if _job_id.get():
        fields['job_id'] = _job_id.get()
    if context.is_valid:
        fields.update(trace_id=format(context.trace_id, '032x'), span_id=format(context.span_id, '016x'))
    body = json.dumps(fields, ensure_ascii=False)
    from opentelemetry._logs import SeverityNumber
    runtime.otel_logger.emit(body=body, severity_number=SeverityNumber.INFO,
                             attributes={'event.name': name})
    _events.info(body, extra={'event_fields': fields})


@contextmanager
def stage(name, **attributes):
    runtime = _runtime
    if runtime is None:
        yield None
        return
    from opentelemetry.trace import Status, StatusCode
    attrs = dict(attributes)
    if _job_id.get():
        attrs['job.id'] = _job_id.get()
    started = time.perf_counter()
    with runtime.tracer.start_as_current_span(name, attributes=attrs,
            record_exception=False, set_status_on_exception=False) as span:
        outcome = 'success'
        result_state = {'failed': False}
        outcome_token = _outcome.set(result_state)
        event('stage.started', stage=name)
        try:
            yield span
        except BaseException as exc:
            outcome = 'error'
            span.set_status(Status(StatusCode.ERROR))
            span.set_attribute('error.type', type(exc).__name__)
            event('stage.failed', stage=name, error_type=type(exc).__name__)
            raise
        finally:
            if result_state['failed']:
                outcome = 'error'
            seconds = time.perf_counter() - started
            labels = {'stage': name, 'outcome': outcome}
            runtime.duration.record(seconds, labels)
            runtime.completed.add(1, labels)
            event('stage.finished', stage=name, outcome=outcome, duration_ms=round(seconds * 1000, 1))
            _outcome.reset(outcome_token)


def mark_failed(span, error_type='HandledError'):
    state = _outcome.get()
    if state is not None:
        state['failed'] = True
    if span is not None:
        from opentelemetry.trace import Status, StatusCode
        span.set_status(Status(StatusCode.ERROR))
        span.set_attribute('error.type', error_type)


def observed_stage(name):
    def decorate(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            with stage(name) as span:
                result = fn(*args, **kwargs)
                if isinstance(result, dict) and result.get('error'):
                    mark_failed(span)
                return result
        return wrapped
    return decorate


def observed_job(kind):
    def decorate(fn):
        @functools.wraps(fn)
        def wrapped(job_id, *args, **kwargs):
            token = _job_id.set(job_id)
            runtime = _runtime
            if runtime:
                runtime.active.add(1, {'kind': kind})
            try:
                with stage('job.' + kind) as span:
                    result = fn(job_id, *args, **kwargs)
                    job = fn.__globals__.get('_jobs', {}).get(job_id, {})
                    organization = (job.get('result') or {}).get('organization_status')
                    if job.get('status') == 'error' or organization in ('error', 'partial'):
                        mark_failed(span)
                    return result
            finally:
                if runtime:
                    runtime.active.add(-1, {'kind': kind})
                _job_id.reset(token)
        return wrapped
    return decorate


def start_thread(target, args):
    context = contextvars.copy_context()
    thread = threading.Thread(target=context.run, args=(target, *args), daemon=True)
    thread.start()
    return thread


def llm_usage(provider, input_tokens=None, output_tokens=None):
    if _runtime:
        for direction, count in [('input', input_tokens), ('output', output_tokens)]:
            if isinstance(count, int) and count >= 0:
                _runtime.tokens.add(count, {'provider': provider, 'direction': direction})
        event('llm.usage', provider=provider, input_tokens=input_tokens, output_tokens=output_tokens)


def quota_wait(provider, seconds):
    if _runtime:
        _runtime.wait.record(seconds, {'provider': provider})
        event('llm.quota_wait', provider=provider, duration_ms=round(seconds * 1000, 1))


class TelemetryMiddleware:
    """Use route templates; never export raw URLs, headers or body."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        runtime = _runtime
        if runtime is None or scope['type'] != 'http':
            return await self.app(scope, receive, send)
        from opentelemetry.trace import SpanKind, Status, StatusCode
        started, status = time.perf_counter(), 500
        with runtime.tracer.start_as_current_span('http.request', kind=SpanKind.SERVER,
                record_exception=False, set_status_on_exception=False) as span:
            async def tracked_send(message):
                nonlocal status
                if message['type'] == 'http.response.start':
                    status = message['status']
                await send(message)
            try:
                await self.app(scope, receive, tracked_send)
            finally:
                route = getattr(scope.get('route'), 'path', 'unmatched')
                method = scope.get('method', 'OTHER')
                method = method if method in {'GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS', 'HEAD'} else 'OTHER'
                span.update_name(method + ' ' + route)
                span.set_attributes({'http.route': route, 'http.request.method': method,
                                     'http.response.status_code': status})
                if status >= 500:
                    span.set_status(Status(StatusCode.ERROR))
                runtime.http_duration.record(time.perf_counter() - started,
                    {'route': route, 'method': method, 'status': str(status)})
