"""Run the local observability stack using native binaries, without Docker."""
import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from scripts.native_observability import ROOT, commands, grafana_environment, install_all, write_configs
from start_app import _stop_process

PORTS = (3000, 3100, 3200, 4318, 9090, 12345, 14318, 9095, 9096, 7946, 7947)


def assert_ports_available():
    for port in PORTS:
        with socket.socket() as sock:
            try:
                sock.bind(('127.0.0.1', port))
            except OSError as exc:
                raise RuntimeError(f'Porta {port} ocupada; pare a stack anterior antes de iniciar.') from exc


def wait_until_ready(processes, state_dir, timeout=120):
    import requests
    urls = ('http://127.0.0.1:9090/-/ready', 'http://127.0.0.1:3100/ready',
            'http://127.0.0.1:3200/ready', 'http://127.0.0.1:12345/-/ready',
            'http://127.0.0.1:3000/api/health')
    pending = set(urls)
    deadline = time.monotonic() + timeout
    while pending and time.monotonic() < deadline:
        for name, process in processes:
            if process.poll() is not None:
                raise RuntimeError(f'{name} encerrou. Consulte {state_dir / "logs" / (name + ".log")}')
        for url in list(pending):
            try:
                with requests.get(url, timeout=1) as response:
                    if response.status_code == 200:
                        pending.remove(url)
            except requests.RequestException:
                continue
        if pending:
            time.sleep(1)
    if pending:
        raise RuntimeError('Serviços não ficaram prontos em 120s. Consulte os logs em ' + str(state_dir / 'logs'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install-only', action='store_true', help='Baixar e verificar os binários sem iniciar')
    parser.add_argument('--smoke', action='store_true', help='Verificar entrega dos três sinais e encerrar')
    parser.add_argument('--state-dir', type=Path, default=ROOT / 'data/observability')
    args = parser.parse_args(argv)
    state_dir = args.state_dir.resolve()
    load_dotenv(ROOT / '.env')
    processes, files = [], []
    try:
        if not args.install_only and not os.getenv('GRAFANA_ADMIN_PASSWORD', '').strip():
            raise RuntimeError('Defina GRAFANA_ADMIN_PASSWORD no .env antes de iniciar.')
        if not args.install_only:
            assert_ports_available()
        binaries = install_all(state_dir)
        configs = write_configs(state_dir)
        if args.install_only:
            print('Binários verificados e configurações preparadas.', flush=True)
            return 0
        (state_dir / 'logs').mkdir(parents=True, exist_ok=True)
        for name, command in commands(binaries, state_dir, configs).items():
            output = (state_dir / 'logs' / (name + '.log')).open('ab')
            files.append(output)
            env = grafana_environment(state_dir, configs) if name == 'grafana' else os.environ.copy()
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
            processes.append((name, process))
            print(f'Iniciando {name}...', flush=True)
        wait_until_ready(processes, state_dir)
        print('Observabilidade pronta. Grafana: http://localhost:3000 (usuário admin)', flush=True)
        print('Logs: ' + str(state_dir / 'logs'), flush=True)
        if args.smoke:
            return subprocess.run([sys.executable, '-m', 'scripts.smoke_observability'],
                                  cwd=ROOT, env=os.environ.copy(), check=False).returncode
        print('Agora execute python start_app.py em outro terminal. Ctrl+C para parar a stack.', flush=True)
        while True:
            for name, process in processes:
                if process.poll() is not None:
                    raise RuntimeError(f'{name} parou. Consulte seu log em {state_dir / "logs"}.')
            time.sleep(0.5)
    except KeyboardInterrupt:
        print('Parando a observabilidade...', flush=True)
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print('Erro: ' + str(exc), flush=True)
        return 1
    finally:
        for _, process in reversed(processes):
            _stop_process(process)
        for file in files:
            file.close()


if __name__ == '__main__':
    sys.exit(main())
