"""Verify real Alloy -> Prometheus/Loki/Tempo delivery through Grafana proxies."""
import os
import time
import uuid
import requests
from app import telemetry


def main():
    password = os.environ['GRAFANA_ADMIN_PASSWORD']
    session = requests.Session()
    session.auth = ('admin', password)
    base = os.getenv('GRAFANA_URL', 'http://127.0.0.1:3000')
    def get(path, params=None):
        response = session.get(base + path, params=params, timeout=5)
        response.raise_for_status()
        return response.json()
    def wait_for(check, label):
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            try:
                if check():
                    print(label + ': OK')
                    return
            except (requests.RequestException, ValueError, KeyError):
                pass
            time.sleep(2)
        raise RuntimeError(label + ' did not become available')
    wait_for(lambda: get('/api/health')['database'] == 'ok', 'Grafana readiness')
    assert {item['uid'] for item in get('/api/datasources')} >= {'prometheus', 'loki', 'tempo'}
    panels = get('/api/dashboards/uid/licitgraphai')['dashboard']['panels']
    assert panels
    os.environ['OBSERVABILITY_ENABLED'] = 'true'
    os.environ['OTEL_TRACES_SAMPLER_ARG'] = '1.0'
    marker = str(uuid.uuid4())
    telemetry.configure()
    try:
        with telemetry.stage('smoke.delivery') as span:
            trace_id = format(span.get_span_context().trace_id, '032x')
            telemetry.event('smoke.delivery', smoke_id=marker)
            telemetry.llm_usage('gemini', 10, 5)
            telemetry.quota_wait('gemini', 0.1)
        for provider in telemetry._runtime.providers:
            assert provider.force_flush(timeout_millis=5000)
        prefix = '/api/datasources/proxy/uid/'
        wait_for(lambda: bool(get(prefix + 'prometheus/api/v1/query',
            {'query': 'licit_operations{stage="smoke.delivery",outcome="success"}'})['data']['result']), 'Prometheus metric')
        for panel in panels:
            if panel['datasource']['uid'] == 'prometheus':
                query = panel['targets'][0]['expr']
                assert get(prefix + 'prometheus/api/v1/query', {'query': query})['status'] == 'success'
        print('Dashboard PromQL: OK')
        wait_for(lambda: bool(get(prefix + 'loki/loki/api/v1/query_range',
            {'query': '{service_name="licitgraphai-api"} |= "' + marker + '"',
             'start': str(time.time_ns() - 600_000_000_000), 'limit': 10})['data']['result']), 'Loki event')
        wait_for(lambda: bool(get(prefix + 'tempo/api/traces/' + trace_id)), 'Tempo trace')
    finally:
        telemetry.shutdown()


if __name__ == '__main__':
    main()
