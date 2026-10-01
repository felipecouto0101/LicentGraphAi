# LicitGraphAi

Aplicação em Python que transforma editais em PDF em um **mapa navegável de temas e subtemas**, com páginas e trechos de origem para conferência.

A interface atual permite enviar o documento, buscar assuntos e consultar suas fontes. A organização usa IA, sem categorias fixas; chat e explicações sob demanda ainda não estão disponíveis.

## Arquitetura e tecnologias

| Componente | Tecnologias | Responsabilidade |
| --- | --- | --- |
| Interface | Streamlit | Upload, busca, navegação por temas, fontes e acompanhamento do progresso. |
| API | FastAPI, Uvicorn e Pydantic | Endpoints, jobs em background e retomada da organização. |
| Processamento do PDF | pdfplumber e LangChain | Extração de texto e tabelas, identificação de seções e divisão em chunks com referências de página. |
| Organização com IA | LangChain: ChatGoogleGenerativeAI (Gemini) e ChatGroq (Groq) | Identificação de subtemas e agrupamento em temas, com saída estruturada. |
| Recuperação de fontes | Python | Validação das referências, recuperação do contexto original e apresentação de parágrafos, listas e tabelas. |
| Pipeline completo | LangGraph, Sentence Transformers e ChromaDB | Orquestração da análise completa, embeddings e recuperação de contexto com RAG. |

**O mapa de assuntos usa o texto dos chunks diretamente, sem embeddings ou ChromaDB.** O pipeline completo com RAG permanece disponível em `/analyze/full`, fora da interface atual.

### Fluxo principal: mapa de assuntos

1. **Upload:** a interface envia o PDF à API, que cria um job e inicia o processamento em background.
2. **Extração:** o leitor preserva páginas, linhas e tabelas; identifica seções e divide o texto em chunks.
3. **Identificação:** a IA recebe o texto em lotes e identifica subtemas, indicando IDs e linhas das fontes.
4. **Validação e agrupamento:** o backend verifica as referências; a IA consolida os assuntos em temas. Lotes inválidos podem ser corrigidos ou subdivididos.
5. **Apresentação:** o backend recupera o contexto no PDF original, deduplica sobreposições e vincula continuações identificáveis. A interface exibe o mapa, as fontes e o progresso.

Se houver uma interrupção, os lotes validados ficam disponíveis como mapa parcial. Quando ainda não há subtemas validados, a interface apresenta o índice original e o motivo da interrupção.

### Fluxo completo: análise com RAG

Disponível pela API, utiliza LangGraph para executar quatro etapas: **leitura e chunking → embeddings e ChromaDB → análise com LLM e contexto recuperado → geração de checklist**. Usa Groq e mantém checkpoints em disco para retomada.

## Executar localmente

Use Python 3.11 ou 3.12. Na raiz do projeto, pelo Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

No Linux/macOS, a ativação é `source .venv/bin/activate`. No PowerShell, use `.\.venv\Scripts\Activate.ps1` e `Copy-Item .env.example .env`.

Configure o mapa com Gemini no `.env`:

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

Esses valores são **orçamentos locais configuráveis**, não cotas universais ou saldo consultado no provedor. Ajuste-os aos limites da sua conta. Mantenha o `.env` fora do Git.

Inicie API e interface:

```bash
python start_app.py
```

- Interface: [localhost:8501](http://localhost:8501)
- Documentação da API: [127.0.0.1:8002/docs](http://127.0.0.1:8002/docs)

Para executar separadamente, use um terminal para cada comando:

```bash
python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8002
python -m streamlit run app/streamlit_app.py
```

### Alternativa: Groq

Configure `TOPIC_LLM_PROVIDER=groq` e `GROQ_API_KEY`. Chaves extras usam `GROQ_API_KEY_2`, `GROQ_API_KEY_3` etc. Para contas de organizações diferentes, `GROQ_INDEPENDENT_ACCOUNTS=true` permite procurar outra conta disponível antes de esperar. O padrão é `false`.

O pipeline completo requer a configuração Groq mesmo quando o mapa usa Gemini. Reinicie o backend após alterar o `.env`.

Ambos os provedores usam LangChain. O cliente Gemini mantém controle local de cotas, tentativas e progresso; as tentativas internas do SDK ficam desativadas. Após atualizar esta versão, execute `python -m pip install -r requirements.txt` para instalar `langchain-google-genai`.

## API e armazenamento

| Endpoint | Função |
| --- | --- |
| `POST /analyze/upload` | Envia o PDF e inicia o mapa; retorna um `job_id`. |
| `GET /job/{job_id}` | Retorna progresso, resultado e estado da organização. |
| `POST /job/{job_id}/organize-map` | Retoma a organização interrompida na mesma sessão. |
| `POST /analyze/full` | Executa o pipeline completo com LangGraph/RAG. |
| `GET /status` | Informa disponibilidade da API e configuração do provedor. |

- **Mapa:** jobs, páginas e cache de lotes ficam em memória. Reiniciar o backend exige reenviar o PDF.
- **Uploads:** arquivos ficam em `data/raw/uploads/`.
- **Pipeline completo:** índice em `data/vector_db/` e checkpoints em `data/checkpoints/`, configuráveis por `CHROMA_PERSIST_DIRECTORY` e `LICIT_CHECKPOINT_DIR`.

## Limitações

PDFs precisam de texto extraível; OCR não está implementado. Colunas, tabelas e continuações com diagramação irregular podem exigir conferência no original. Referências validadas comprovam a origem do texto, mas não garantem que todos os assuntos foram identificados ou que os agrupamentos estejam perfeitos.

O tempo depende do documento, das respostas e das cotas da API. O controle preventivo é local e não acompanha consumo externo; esperas e falhas são exibidas no progresso. A recuperação e a apresentação das fontes não exigem chamadas adicionais à IA.

## Testes

Execute as suítes do mapa, cliente Gemini e fontes, sem credenciais reais:

```bash
python -m unittest discover -s tests -p 'test_topic*.py' -v
python -m unittest discover -s tests -p 'test_gemini*.py' -v
python -m unittest discover -s tests -p 'test_source*.py' -v
python -m unittest discover -s tests -p 'test_pdf_tables.py' -v
```

Os testes cobrem referências, agrupamento, cache, resultados parciais, cotas, retries, contexto original, tabelas e continuações entre páginas.
