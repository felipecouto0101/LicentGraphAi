"""Install verified upstream binaries and generate localhost-only native configs."""
import hashlib
import json
import os
import platform
import shutil
import stat
import tarfile
import tempfile
import requests
from urllib.parse import urlparse
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def platform_key():
    system, machine = platform.system(), platform.machine().lower()
    if system not in ('Windows', 'Linux') or machine not in ('amd64', 'x86_64'):
        raise RuntimeError('A stack nativa suporta Windows/Linux x64.')
    return system.lower() + '-amd64'


def checksum(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def extract_archive(archive, directory):
    """Reject traversal, links and special files before extracting regular files."""
    def safe(name):
        path = PurePosixPath(name.replace('\\', '/'))
        if path.is_absolute() or '..' in path.parts or ':' in name:
            raise ValueError('Archive contains an unsafe path')
    def copy_file(name, stream):
        destination = directory.joinpath(*PurePosixPath(name.replace('\\', '/')).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('wb') as output:
            shutil.copyfileobj(stream, output)
    if archive.name.endswith('.zip'):
        with zipfile.ZipFile(archive) as source:
            for item in source.infolist():
                safe(item.filename)
                if stat.S_ISLNK(item.external_attr >> 16):
                    raise ValueError('Archive contains a symbolic link')
            for item in source.infolist():
                if not item.is_dir():
                    with source.open(item) as stream:
                        copy_file(item.filename, stream)
    else:
        with tarfile.open(archive, 'r:gz') as source:
            for item in source.getmembers():
                safe(item.name)
                if not (item.isdir() or item.isfile()):
                    raise ValueError('Archive contains a link or special file')
            for item in source.getmembers():
                if item.isfile():
                    with source.extractfile(item) as stream:
                        copy_file(item.name, stream)


def install_binary(name, spec, state_dir):
    url = urlparse(spec['url'])
    if url.scheme != 'https' or url.hostname not in ('github.com', 'dl.grafana.com'):
        raise ValueError('Only HTTPS downloads from official release hosts are allowed')
    target = state_dir / 'bin' / platform_key() / name
    marker = target / 'installed.json'
    if marker.exists():
        record = json.loads(marker.read_text())
        executable = target / record['executable']
        if (record['archive_sha256'] == spec['sha256'] and executable.is_file()
                and checksum(executable) == record['executable_sha256']):
            return executable
    cache = state_dir / 'downloads'
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / spec['url'].rsplit('/', 1)[-1]
    if not archive.exists() or checksum(archive) != spec['sha256']:
        print(f'Baixando {name} (primeira instalação)...', flush=True)
        temporary = archive.with_suffix(archive.suffix + '.part')
        try:
            with requests.get(spec['url'], stream=True, timeout=(10, 60)) as response:
                response.raise_for_status()
                with temporary.open('wb') as file:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        file.write(block)
            if checksum(temporary) != spec['sha256']:
                raise ValueError(f'SHA256 inválido para {name}; arquivo não instalado.')
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as temp:
        unpacked = Path(temp)
        extract_archive(archive, unpacked)
        matches = list(unpacked.rglob(spec['executable']))
        if len(matches) != 1:
            raise ValueError(f'Executável de {name} não encontrado de forma única.')
        executable = matches[0]
        executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
        record = {'archive_sha256': spec['sha256'], 'executable': str(executable.relative_to(unpacked)),
                  'executable_sha256': checksum(executable)}
        (unpacked / 'installed.json').write_text(json.dumps(record))
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(unpacked, target)
    return target / record['executable']


def install_all(state_dir):
    manifest = json.loads((ROOT / 'observability/native-binaries.json').read_text())
    return {name: install_binary(name, spec, state_dir)
            for name, spec in manifest[platform_key()].items()}


def write_configs(state_dir):
    """Use POSIX paths even on Windows; YAML/Alloy avoid escaped backslashes."""
    configs = state_dir / 'config'
    configs.mkdir(parents=True, exist_ok=True)
    for service in ('alloy', 'prometheus', 'loki', 'tempo', 'grafana'):
        (state_dir / 'data' / service).mkdir(parents=True, exist_ok=True)
    for name in ('loki', 'tempo'):
        data = (state_dir / 'data' / name).as_posix()
        text = (ROOT / 'observability' / (name + '.yaml')).read_text()
        text = text.replace('/var/tempo', data).replace('/loki', data)
        text = text.replace('  http_listen_port:', '  http_listen_address: 127.0.0.1\n  http_listen_port:')
        grpc_port = 9095 if name == 'loki' else 9096
        text = text.replace('server:\n', f'server:\n  grpc_listen_address: 127.0.0.1\n  grpc_listen_port: {grpc_port}\n', 1)
        memberlist_port = 7946 if name == 'loki' else 7947
        text += f'\nmemberlist:\n  bind_addr: [127.0.0.1]\n  bind_port: {memberlist_port}\n'
        text = text.replace('0.0.0.0:4318', '127.0.0.1:14318')
        if name == 'loki':
            text = text.replace('common:\n', 'common:\n  instance_addr: 127.0.0.1\n', 1)
        (configs / (name + '.yaml')).write_text(text)
    text = (ROOT / 'observability/config.alloy').read_text()
    for before, after in [('0.0.0.0:4318', '127.0.0.1:4318'), ('tempo:4318', '127.0.0.1:14318'),
                          ('loki:3100', '127.0.0.1:3100'), ('prometheus:9090', '127.0.0.1:9090')]:
        text = text.replace(before, after)
    (configs / 'config.alloy').write_text(text)
    text = (ROOT / 'observability/prometheus.yaml').read_text()
    text = text.replace('/etc/prometheus/alerts.yaml', (ROOT / 'observability/alerts.yaml').as_posix())
    text = text.replace('alloy:12345', '127.0.0.1:12345')
    (configs / 'prometheus.yaml').write_text(text)
    provisioning = configs / 'provisioning'
    shutil.copytree(ROOT / 'observability/grafana/provisioning', provisioning, dirs_exist_ok=True)
    source = provisioning / 'datasources/sources.yaml'
    text = source.read_text()
    for name, port in [('prometheus', 9090), ('loki', 3100), ('tempo', 3200)]:
        text = text.replace(f'{name}:{port}', f'127.0.0.1:{port}')
    source.write_text(text)
    source = provisioning / 'dashboards/default.yaml'
    source.write_text(source.read_text().replace('/var/lib/grafana/dashboards',
                     (ROOT / 'observability/grafana/dashboards').as_posix()))
    (configs / 'grafana.ini').write_text('[server]\nhttp_addr = 127.0.0.1\nhttp_port = 3000\n')
    return configs


def commands(binaries, state_dir, configs):
    def data(name):
        return str(state_dir / 'data' / name)
    return {
        'prometheus': [str(binaries['prometheus']), '--config.file=' + str(configs / 'prometheus.yaml'),
                       '--web.listen-address=127.0.0.1:9090', '--web.enable-remote-write-receiver',
                       '--storage.tsdb.path=' + data('prometheus'), '--storage.tsdb.retention.time=7d'],
        'loki': [str(binaries['loki']), '-config.file=' + str(configs / 'loki.yaml')],
        'tempo': [str(binaries['tempo']), '-target=all', '-config.file=' + str(configs / 'tempo.yaml')],
        'alloy': [str(binaries['alloy']), 'run', '--server.http.listen-addr=127.0.0.1:12345',
                  '--storage.path=' + data('alloy'), str(configs / 'config.alloy')],
        'grafana': [str(binaries['grafana']), 'server', '--homepath=' + str(binaries['grafana'].parent.parent),
                    '--config=' + str(configs / 'grafana.ini')],
    }


def grafana_environment(state_dir, configs):
    env = os.environ.copy()
    env.update(GF_PATHS_DATA=str(state_dir / 'data/grafana'),
               GF_PATHS_LOGS=str(state_dir / 'logs'), GF_PATHS_PLUGINS=str(state_dir / 'plugins'),
               GF_PATHS_PROVISIONING=str(configs / 'provisioning'),
               GF_SECURITY_ADMIN_PASSWORD=os.environ['GRAFANA_ADMIN_PASSWORD'],
               GF_USERS_ALLOW_SIGN_UP='false', GF_ANALYTICS_REPORTING_ENABLED='false',
               GF_ANALYTICS_CHECK_FOR_UPDATES='false')
    return env
