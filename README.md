# LicentGraphAi

Aplicação para organizar informações de editais em PDF. Lê o arquivo, extrai itens com a Groq, apresenta explicações em linguagem simples e separa documentos de participação, atividades da execução e anexos de consulta. **O resultado é uma análise automática para conferência, não substitui a leitura do edital.**

## Início rápido

Requer Python 3.11 ou 3.12 e uma chave da Groq para análise real. Na raiz do projeto, no PowerShell:

```powershell
git clone https://github.com/felipecouto0101/LicentGraphAi.git
cd LicentGraphAi
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edite o `.env` e substitua `your_groq_api_key_here` pela sua chave em `GROQ_API_KEY`. Depois inicie os dois serviços:

```powershell
python start_app.py
```

Abra **http://localhost:8501** para usar a interface. A documentação da API fica em **http://127.0.0.1:8002/docs**. O primeiro início pode demorar enquanto o modelo de embeddings é carregado.

Se preferir terminais separados, inicie a API e depois o Streamlit:

```powershell
python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8002
python -m streamlit run app/streamlit_app.py
```

No Linux ou macOS, ative o ambiente com `source .venv/bin/activate` e crie o `.env` com `cp .env.example .env`.

## Configuração

| Variável | Uso |
| --- | --- |
| `GROQ_API_KEY` | Obrigatória para análise real. |
| `GROQ_API_KEY_2` a `GROQ_API_KEY_5` | Chaves adicionais opcionais; a rotação é usada quando uma chave esgota a cota diária. |
| `NODE3_MOCK_MODE` | `false` por padrão; `true` gera dados de demonstração sem chamar a Groq. |
| `CHROMA_PERSIST_DIRECTORY` | Diretório do banco vetorial; padrão `./data/vector_db`. |
| `LICIT_CHECKPOINT_DIR` | Diretório de checkpoints locais; padrão `./data/checkpoints`. |

Copie `.env.example` para `.env`; o arquivo `.env` é ignorado pelo Git. O modelo da Groq e o modelo de embeddings são definidos no código dos respectivos nós. As variáveis antigas `DEFAULT_MODEL`, `CHROMA_COLLECTION_NAME` e `API_PORT` não controlam este fluxo.

Para conferir o modo ativo, abra `http://127.0.0.1:8002/status`. O modo de demonstração é sinalizado na interface e não deve ser usado como análise de um edital real.

## Como a análise funciona

1. **Nó 1 — PDF:** extrai texto com `pdfplumber` e divide o conteúdo em trechos, preservando o número físico da página. PDFs sem texto extraível podem exigir OCR antes do uso.
2. **Nó 2 — embeddings:** codifica os trechos com `sentence-transformers` e cria uma coleção ChromaDB separada para cada execução.
3. **Nó 3 — extração e explicação:** envia os trechos em lotes à Groq; busca trechos relacionados no ChromaDB para ajudar na conferência; escreve resumo e requisitos em parágrafos; depois explica **cada item extraído** de participação e seleção em lotes de cinco. Verifica IDs e repete apenas os faltantes até duas vezes. Se ainda houver lacunas, o job termina com erro.
4. **Nó 4 — checklist:** organiza os candidatos a documentos para proposta ou habilitação. Documentos da execução, anexos fornecidos pelo órgão e itens de etapa incerta ficam separados. Se o nó 3 falhar, o nó 4 é ignorado.

O resumo usa uma amostra dos itens extraídos e informa sua cobertura. As explicações item a item percorrem todos os itens **identificados pela extração**. Isso não garante que todas as cláusulas do PDF tenham sido encontradas. Trechos recuperados por similaridade são pistas para verificação; não provam automaticamente cada conclusão.

### Abas do resultado

| Aba | Conteúdo |
| --- | --- |
| Resumo | Visão geral, riscos, alguns prazos e trechos do PDF com página para conferência. |
| Perguntas respondidas | Itens extraídos por tema e trechos relacionados. |
| Análise de requisitos | Condições de participação, disputa, julgamento, prazos e explicações item a item. |
| Checklist de documentos | Candidatos a documentos da proposta/habilitação, com duplicatas comuns removidas. A obrigatoriedade aparece como **A confirmar** quando não há evidência específica. |
| Execução e anexos | Atividades e entregas, documentos após a contratação, anexos para consulta e itens de etapa incerta. |
| JSON completo | Dados brutos da análise, para diagnóstico. |

## Progresso, limites e retomada

Após o upload, a interface acompanha um job em segundo plano. A barra indica **lotes concluídos na etapa atual**; não é uma porcentagem global. A API também informa o estado em `GET /job/{job_id}`. `GET /status` verifica a configuração do serviço, não o progresso do job.

A Groq pode responder `429 Too Many Requests`. O cliente tenta novamente e há esperas entre lotes, mas o intervalo fixo não garante disponibilidade da cota de requisições ou de tokens. Uma falha definitiva aparece no job. Não há tentativa infinita de completar uma explicação.

Após cada lote de extração e cada explicação válida, o nó 3 salva um checkpoint em `data/checkpoints/`. Se você iniciar **outra análise do mesmo PDF** com o mesmo modelo e a mesma versão do pipeline, ele reaproveita as chamadas concluídas e continua nos itens faltantes. A leitura do PDF e a indexação vetorial são executadas novamente. Se o conteúdo, o modelo ou a versão do checkpoint mudar, a extração começa de novo. Jobs da API existem apenas na memória do processo; o ID antigo deixa de funcionar após reiniciar a API, mas os checkpoints permanecem no disco.

Os checkpoints podem conter conteúdo extraído do edital, ficam só na sua máquina e são ignorados pelo Git. Para forçar uma análise completamente nova, limpe `data/checkpoints/` antes de enviar o PDF; isso também descarta a retomada de outros editais. As coleções ChromaDB por execução persistem no disco e ainda não possuem limpeza automática.

## Uso direto em Python

```python
from dotenv import load_dotenv
load_dotenv()

from app.rag.langgraph_workflow import run_licit_graph_pipeline

state = run_licit_graph_pipeline("data/raw/uploads/edital.pdf")
if state.get("error"):
    raise RuntimeError(state["error"])

print(state["analysis"]["explanatory_report"])
print(state["checklist"]["resumo"])
```

## Testes

```powershell
python -m pytest tests/ -v
```

Há testes unitários com respostas simuladas para extração, páginas, classificação, explicações e retomada de checkpoints. Eles não comprovam a qualidade da redação nem a correção das exigências de um edital real; valide um PDF conhecido antes de usar o relatório para tomar decisões.

## Estrutura principal

```text
app/api/main.py                  API FastAPI e jobs
app/rag/pdf_reader.py            Extração de texto do PDF
app/rag/node_1_reader_chunker.py Páginas e trechos
app/rag/node_2_embeddings.py     Embeddings e ChromaDB
app/rag/node_3_analyzer.py       Extração, RAG, explicações e checkpoints
app/rag/node_4_document_generator.py Checklist
app/rag/langgraph_workflow.py    Orquestração
app/streamlit_app.py             Interface
start_app.py                     Inicia API e interface
tests/                           Testes
```
