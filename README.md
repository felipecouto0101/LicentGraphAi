# LicitGraphAi

Aplicação em Python que organiza editais em PDF em um **mapa navegável de temas e subtemas**, com páginas e trechos do documento original.

Pela interface, o usuário envia um PDF, acompanha o processamento, busca assuntos e consulta as fontes. A IA identifica e agrupa os assuntos conforme o conteúdo de cada edital, sem categorias fixas.

## Arquitetura e tecnologias

| Camada | Tecnologias | Responsabilidade |
| --- | --- | --- |
| Interface | Streamlit | Upload, busca, navegação por temas e subtemas, fontes e progresso. |
| API | FastAPI, Uvicorn e Pydantic | Endpoints, validação das requisições e jobs em background. |
| Leitura de documentos | pdfplumber e LangChain | Extração de texto e tabelas, identificação de seções e divisão em chunks com referências de página. |
| Organização com IA | LangChain, Gemini e Groq | Identificação de subtemas e agrupamento em temas com respostas estruturadas. |
| Recuperação de fontes | Python | Validação das referências, recuperação de contexto e apresentação de parágrafos, listas e tabelas. |
| Análise completa pela API | LangGraph, Sentence Transformers e ChromaDB | Orquestração do pipeline, embeddings, recuperação de contexto com RAG e geração de checklist. |

Gemini utiliza `ChatGoogleGenerativeAI`; Groq utiliza `ChatGroq`. O mapa de assuntos processa diretamente o texto dos chunks. Embeddings e ChromaDB fazem parte do pipeline de análise completa.

## Fluxo da aplicação

1. **Envio:** a interface envia o PDF à API, que cria um job de processamento.
2. **Leitura:** o backend extrai texto e tabelas, identifica seções e gera chunks com referências ao documento.
3. **Identificação:** a IA identifica subtemas em lotes e associa cada assunto às fontes.
4. **Organização:** o backend valida as referências e a IA consolida os assuntos em temas.
5. **Consulta:** a interface apresenta o mapa e permite buscar assuntos e abrir os trechos de origem.

O processamento valida as fontes, elimina sobreposições e recupera continuações identificáveis entre páginas. Respostas inválidas ou truncadas passam por correção ou subdivisão dos lotes.

Em caso de interrupção, o mapa apresenta os lotes validados e permite retomar a organização na mesma sessão. Quando não há subtemas validados, apresenta as seções originais do PDF e o motivo da interrupção.

A análise completa, acessível por `/analyze/full`, utiliza quatro etapas no LangGraph: **leitura e chunking → embeddings e ChromaDB → análise com Groq e RAG → checklist**.

## Executar localmente

Use Python 3.11 ou 3.12. Na raiz do projeto, pelo Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

No Linux/macOS, ative com `source .venv/bin/activate`. No PowerShell, use `.\.venv\Scripts\Activate.ps1` e `Copy-Item .env.example .env`.

Configure o provedor no `.env`:

```dotenv
TOPIC_LLM_PROVIDER=gemini
GEMINI_API_KEY=sua_chave
GEMINI_MODEL=gemini-3.5-flash-lite
NODE3_MOCK_MODE=false

GEMINI_RPM_BUDGET=10
GEMINI_TPM_BUDGET=100000
GEMINI_RPD_BUDGET=500
GEMINI_OUTPUT_TOKEN_BUDGET=8192
```

Os orçamentos controlam o uso local de requisições e tokens. Configure-os conforme os limites da conta; o controle local não consulta saldo nem acompanha consumo externo. Mantenha o `.env` fora do Git.

Inicie os serviços:

```bash
python start_app.py
```

- **Interface:** [localhost:8501](http://localhost:8501)
- **Documentação da API:** [127.0.0.1:8002/docs](http://127.0.0.1:8002/docs)

Para executar separadamente, use um terminal por serviço:

```bash
python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8002
python -m streamlit run app/streamlit_app.py
```

### Configuração Groq

Use `TOPIC_LLM_PROVIDER=groq` e `GROQ_API_KEY`. Chaves adicionais usam `GROQ_API_KEY_2`, `GROQ_API_KEY_3` etc.

Para contas de organizações diferentes, `GROQ_INDEPENDENT_ACCOUNTS=true` permite selecionar uma conta disponível antes de esperar. O padrão é `false`. O pipeline de análise completa requer Groq mesmo quando o mapa utiliza Gemini.

Reinicie o backend após alterar o `.env`.

As origens CORS permitidas são `http://localhost:8501` e `http://127.0.0.1:8501`.
Use `API_CORS_ORIGINS`, com URLs separadas por vírgula, para configurar outras origens.

## Execução e acompanhamento

A interface verifica a disponibilidade da API em background e reutiliza o resultado por 15 segundos. Fragmentos do Streamlit atualizam conexão e progresso; o conteúdo de temas e fontes é renderizado ao abrir seus respectivos painéis.

A API carrega leitores, clientes de IA e o pipeline completo conforme a execução dos jobs. O endpoint `/status` verifica a configuração sem carregar modelos ou processar PDFs.

Os logs registram tempos de renderização, verificação da API e carregamento do backend: `ui_render`, `ui_api_health`, `api_startup` e `api_full_pipeline_import`. Esperas por cotas e interrupções aparecem no progresso.

## API e armazenamento

| Endpoint | Função |
| --- | --- |
| `POST /analyze/upload` | Envia o PDF e inicia o mapa; retorna um `job_id`. |
| `GET /job/{job_id}` | Consulta progresso, resultado e estado da organização. |
| `POST /job/{job_id}/organize-map` | Retoma a organização na mesma sessão. |
| `POST /analyze/full` | Executa a análise completa com LangGraph e RAG. |
| `GET /status` | Consulta disponibilidade da API e configuração do provedor. |

- **Mapa:** jobs, páginas e cache dos lotes ficam em memória. Após reiniciar o backend, o PDF precisa ser reenviado.
- **Uploads:** `data/raw/uploads/`.
- **Análise completa:** índice em `data/vector_db/` e checkpoints em `data/checkpoints/`, configuráveis por `CHROMA_PERSIST_DIRECTORY` e `LICIT_CHECKPOINT_DIR`.

## Escopo de processamento

A leitura trabalha com PDFs que contêm texto extraível. A fidelidade de tabelas, colunas e continuações depende da diagramação do documento. As referências permitem conferir a origem das informações; a cobertura e o agrupamento dos assuntos dependem da identificação pela IA.

O tempo de processamento varia conforme o tamanho do edital, as respostas do modelo e as cotas do provedor. A recuperação e a apresentação das fontes usam o texto extraído, sem chamadas adicionais à IA.

## Testes

As suítes usam respostas simuladas, sem credenciais reais:

```bash
python -m unittest discover -s tests -p 'test_topic*.py' -v
python -m unittest discover -s tests -p 'test_gemini*.py' -v
python -m unittest discover -s tests -p 'test_source*.py' -v
python -m unittest discover -s tests -p 'test_pdf_tables.py' -v
python -m unittest discover -s tests -p 'test_groq_account_scheduler.py' -v
python -m unittest discover -s tests -p 'test_ui_runtime.py' -v
python -m unittest discover -s tests -p 'test_api_startup.py' -v
```

A cobertura inclui organização de assuntos, referências, cache, resultados parciais, cotas, tentativas, fontes, tabelas, continuações entre páginas e inicialização dos serviços.

## Integração contínua

O GitHub Actions executa Ruff, testes em Python 3.12, SAST com Semgrep e Bandit,
auditoria de dependências com pip-audit e detecção de segredos com Gitleaks.
Os relatórios ficam disponíveis nos artefatos da execução. O check `CI gate`
consolida o resultado das verificações.

Consulte [a configuração e os critérios de aprovação](docs/ci.md).

O ChromaDB usa o backend Rust local com embeddings explícitos. Os alertas de
servidor Python têm revisão de aplicabilidade com prazo de validade, publicada
junto ao relatório bruto do scanner. Veja [os controles e limites dessa revisão](docs/chroma-security.md).
