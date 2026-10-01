# Observabilidade

O backend gera métricas, eventos estruturados e traces com OpenTelemetry. Alloy
recebe OTLP/HTTP e encaminha métricas ao Prometheus, logs ao Loki e traces ao
Tempo. Grafana contém as três fontes de dados e o dashboard **LicitGraphAi — Operação**.
A infraestrutura é local, com armazenamento persistente, versões fixas e
portas publicadas somente em `127.0.0.1`.

## Executar sem Docker (Windows ou Linux x64)

Na raiz do projeto:

```bash
pip install -r requirements.txt
```

Configure no `.env` uma senha própria em `GRAFANA_ADMIN_PASSWORD` e:

```dotenv
OBSERVABILITY_ENABLED=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318
OTEL_TRACES_SAMPLER_ARG=1.0
APP_ENV=local
```

No primeiro terminal:

```bash
python start_observability.py
```

O iniciador baixa binários oficiais de versões fixas, verifica SHA256, prepara as
configurações locais e inicia os cinco serviços. Não requer Docker, WSL nem
instalação como serviço/administrador. A primeira execução precisa de internet e
baixa arquivos grandes; nas próximas execuções reutiliza a instalação verificada.
Os arquivos ficam em `data/observability/`, ignorados pelo Git. Os hashes e URLs
estão em `observability/native-binaries.json`.

Depois da mensagem **Observabilidade pronta**, execute em outro terminal:

```bash
python start_app.py
```

Abra http://localhost:3000 e entre com usuário `admin` e a senha configurada.
Em **Dashboards → LicitGraphAi**, abra **LicitGraphAi — Operação**. Após enviar um
PDF, os primeiros dados aparecem em cerca de 10–30 segundos. As consultas de
variação precisam de pelo menos duas amostras; painéis sem eventos ficam vazios.

`Ctrl+C` no primeiro terminal encerra os serviços iniciados por ele e mantém os
dados. Se um serviço falhar, o iniciador encerra os demais e informa o arquivo de
log correspondente em `data/observability/logs/`. Portas ocupadas são detectadas
antes do início; pare uma stack anterior ou outro serviço usando as mesmas portas.
A aplicação e a observabilidade têm terminais independentes.

Para apenas baixar/verificar os binários:

```bash
python start_observability.py --install-only
```

Para iniciar a stack, verificar dashboard/consultas e entrega de uma métrica, um
log e um trace, encerrando ao terminar, sem chamar uma IA:

```bash
python start_observability.py --smoke
```

Esse comando carrega a senha do `.env`; funciona no Git Bash e PowerShell.
Windows e Linux são testados pelo CI com os binários reais.

## Execução alternativa com Docker

O Compose permanece disponível para quem usa Docker com containers Linux:

```bash
docker compose -f compose.observability.yaml up -d
python start_app.py
```

Use a mesma configuração do `.env`. Não execute a stack nativa e a stack Docker
ao mesmo tempo: elas usam as mesmas portas públicas.

## O que está instrumentado

- HTTP: duração, método, código de resposta e modelo da rota.
- Jobs: preparo do mapa, organização, retomada e pipeline completo, com contagem
  de execuções ativas e resultado real, incluindo interrupções e mapas parciais.
- Mapa: leitura do PDF, índice estrutural e organização com IA.
- LangGraph: leitura, embeddings/ChromaDB, análise e checklist.
- Gemini e Groq: cada tentativa de chamada, duração, falhas, tokens de entrada e
  saída reportados pelo provedor e tempo gasto em pausas locais.

Os spans `llm.request` medem a chamada ao provedor; `llm.wait` mede a espera.
Contadores de tokens são consumo observado, não custo financeiro nem saldo
oficial de cota. A quantidade de tokens não está disponível quando o provedor
recusa uma chamada antes de devolver uso. Cache não gera novas chamadas.

O `job_id` aparece nos eventos e no atributo `job.id` dos traces, sem virar
label de métricas. No Explore/Loki, pesquise:

```logql
{service_name="licitgraphai-api"} | json | job_id="ID_DO_JOB"
```

O campo `trace_id` liga o log ao Tempo. No Explore/Tempo também é possível
pesquisar `{ resource.service.name = "licitgraphai-api" && span.job.id = "ID_DO_JOB" }`.
Threads de background herdam o contexto do upload. Uma retomada tem um novo
trace, mantendo o mesmo ID do job.

## Limites e operação

A coleta é opcional e desativada por padrão. Não há handshake com Alloy durante
uma requisição. Exportação usa threads e filas limitadas; se o coletor cair,
a análise continua e telemetria pode ser perdida. SDK/exportadores só são
carregados quando a coleta é habilitada. `OTEL_TRACES_SAMPLER_ARG` controla a
fração de traces coletados entre 0 e 1; métricas continuam sendo registradas.

Somente eventos explícitos da instrumentação são enviados ao Loki. Logs brutos
existentes continuam no terminal. Prompts, respostas, conteúdo do PDF, nomes de
arquivos, cabeçalhos, credenciais e mensagens completas de exceção não são
exportados. Tipos de erro e estados operacionais são registrados.

Prometheus e Loki mantêm dados por sete dias. Tempo e os demais serviços usam armazenamento
local; acompanhe o espaço em disco e remova os dados apenas quando quiser apagar
a telemetria. A configuração local não inclui armazenamento em nuvem, autenticação
entre coletores nem alta disponibilidade. As regras `AnalysisFailures` e
`CollectorUnavailable` ficam em Prometheus/Alerts; não há envio de notificações.

```bash
docker compose -f compose.observability.yaml ps
docker compose -f compose.observability.yaml logs --tail=100 alloy tempo loki
docker compose -f compose.observability.yaml down
```

`down` mantém os dados. Alterar a senha no `.env` após a primeira inicialização
não troca a senha armazenada pelo Grafana; use a interface de administração.
O CI testa a stack nativa no Windows e Linux, além do Compose, validando entrega
dos três sinais e consultas do dashboard. Falhas na infraestrutura reprovam o `CI gate`.
