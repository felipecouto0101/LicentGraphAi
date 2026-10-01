"""No-download tests for archive safety, integrity, native configs and cleanup."""
import json
import tarfile
import zipfile
from types import SimpleNamespace
import pytest
from scripts import native_observability as native
import start_observability as launcher


@pytest.mark.parametrize('filename', ['../outside', '/absolute', 'C:/outside', '..\\outside'])
def test_zip_traversal_is_rejected(tmp_path, filename):
    archive = tmp_path / 'archive.zip'
    with zipfile.ZipFile(archive, 'w') as file:
        file.writestr(filename, 'payload')
    with pytest.raises(ValueError, match='unsafe path'):
        native.extract_archive(archive, tmp_path / 'target')


def test_tar_links_are_rejected(tmp_path):
    archive = tmp_path / 'archive.tar.gz'
    with tarfile.open(archive, 'w:gz') as file:
        item = tarfile.TarInfo('link')
        item.type, item.linkname = tarfile.SYMTYPE, '../outside'
        file.addfile(item)
    with pytest.raises(ValueError, match='link or special'):
        native.extract_archive(archive, tmp_path / 'target')


def test_hash_mismatch_never_extracts_or_installs(tmp_path, monkeypatch):
    monkeypatch.setattr(native, 'platform_key', lambda: 'linux-amd64')
    class Download:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def raise_for_status(self): pass
        def iter_content(self, **kwargs): return [b'wrong']
    monkeypatch.setattr(native.requests, 'get', lambda *a, **kw: Download())
    spec = {'url': 'https://github.com/example/binary.zip', 'sha256': '0' * 64, 'executable': 'binary'}
    with pytest.raises(ValueError, match='SHA256'):
        native.install_binary('test', spec, tmp_path)
    assert not (tmp_path / 'bin').exists()
    assert not list(tmp_path.rglob('*.part'))


def test_verified_install_reuses_cache_and_repairs_changed_executable(tmp_path, monkeypatch):
    monkeypatch.setattr(native, 'platform_key', lambda: 'linux-amd64')
    cache = tmp_path / 'downloads'
    cache.mkdir()
    archive = cache / 'binary.zip'
    with zipfile.ZipFile(archive, 'w') as file:
        file.writestr('folder/bin/binary', 'original')
        file.writestr('packaging/wrappers/binary', 'wrapper')
        file.writestr('assets/binary/readme.txt', 'directory with same name')
    spec = {'url': 'https://github.com/example/binary.zip', 'sha256': native.checksum(archive), 'executable': 'bin/binary'}
    def no_network(*args, **kwargs):
        raise AssertionError('Cached archive must not be downloaded again')
    monkeypatch.setattr(native.requests, 'get', no_network)
    executable = native.install_binary('test', spec, tmp_path)
    assert executable.read_text() == 'original'
    assert native.install_binary('test', spec, tmp_path) == executable
    executable.write_text('changed')
    assert native.install_binary('test', spec, tmp_path).read_text() == 'original'


def test_configs_are_localhost_only_and_preserve_dashboard_links(tmp_path):
    configs = native.write_configs(tmp_path / 'state with spaces')
    texts = '\n'.join(p.read_text() for p in configs.rglob('*') if p.is_file())
    assert '0.0.0.0' not in texts
    assert 'tempo:4318' not in texts
    assert '127.0.0.1:14318' in texts
    assert 'http_listen_address: 127.0.0.1' in texts
    assert 'matcherRegex' in texts
    assert 'state with spaces' in texts
    assert '/var/lib/grafana/dashboards' not in texts


def test_manifest_pins_both_platforms_and_strong_digests():
    manifest = json.loads((native.ROOT / 'observability/native-binaries.json').read_text())
    assert set(manifest) == {'windows-amd64', 'linux-amd64'}
    for tools in manifest.values():
        assert set(tools) == {'alloy', 'prometheus', 'loki', 'tempo', 'grafana'}
        for entry in tools.values():
            assert len(entry['sha256']) == 64
            assert entry['url'].startswith(('https://github.com/', 'https://dl.grafana.com/'))


def test_startup_failure_stops_every_started_process(tmp_path, monkeypatch):
    monkeypatch.setenv('GRAFANA_ADMIN_PASSWORD', 'ephemeral-test')
    monkeypatch.setattr(launcher, 'assert_ports_available', lambda: None)
    monkeypatch.setattr(launcher, 'install_all', lambda state: {})
    monkeypatch.setattr(launcher, 'write_configs', lambda state: tmp_path)
    monkeypatch.setattr(launcher, 'commands', lambda *args: {'loki': ['fake'], 'tempo': ['fake']})
    processes, stopped = [], []
    def start(*args, **kwargs):
        process = SimpleNamespace(poll=lambda: None)
        processes.append(process)
        return process
    monkeypatch.setattr(launcher.subprocess, 'Popen', start)
    def fail(*args):
        raise RuntimeError('Service did not become ready')
    monkeypatch.setattr(launcher, 'wait_until_ready', fail)
    monkeypatch.setattr(launcher, '_stop_process', stopped.append)
    assert launcher.main(['--state-dir', str(tmp_path)]) == 1
    assert stopped == list(reversed(processes))
