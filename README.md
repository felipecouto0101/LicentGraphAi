# LicitGraphAi — Analisador Autônomo de Editais de Licitação

Sistema inteligente para análise automática de editais de licitação pública (Lei 14.133/2021). Faz upload de um PDF, extrai o texto, gera embeddings, consulta o banco vetorial via RAG e usa IA para responder perguntas objetivas sobre o edital.

---

## Como funciona

```
PDF → Extração de texto → Chunks → Embeddings (ChromaDB)
                                          ↓
                              Busca vetorial por 8 queries temáticas
                                          ↓
                              Chunks relevantes → LLM (Groq)
                                          ↓
                              Relatório estruturado por pergunta
```

### Pipeline de nós (LangGraph)

| Nó | Responsabilidade |
|----|-----------------|
| **Nó 1** | Lê o PDF, limpa o texto e fragmenta em chunks por seção |
| **Nó 2** | Gera embeddings com `sentence-transformers` e armazena no ChromaDB |
| **Nó 3** | Consulta RAG: busca os chunks mais relevantes por query temática e envia ao LLM |
| **Nó 4** | Categoriza os documentos exigidos (habilitação, fiscal, técnica, etc.) em checklist |

### Perguntas respondidas pelo RAG

1. Posso participar? *(requisitos de habilitação)*
2. Quais são os prazos?
3. Quanto custa e como pago?
4. Como será a seleção ou avaliação?
5. O que devo entregar ou produzir?
6. Quais são as regras de eliminação?
7. Quais documentos são exigidos?
8. Pontos críticos e riscos

Para cada pergunta, o sistema busca os chunks mais relevantes no ChromaDB por similaridade vetorial e os envia ao LLM com um prompt focado — sem processar o edital inteiro de uma vez.

---

## Stack

- **Python 3.11+**
- **LangGraph** — orquestração do pipeline com estado compartilhado
- **LangChain** — chunking e integração com modelos
- **pdfplumber** — extração de texto de PDFs
- **sentence-transformers** (`all-MiniLM-L6-v2`) — geração de embeddings
- **ChromaDB** — banco vetorial para busca semântica
- **Groq API** + **Qwen 3.8 27B** — modelo LLM para análise
- **FastAPI** — API REST para upload e processamento
- **Streamlit** — interface web

---

## Instalação

```bash
# 1. Clone o repositório
git clone <repo>
cd LicitGraphAi

# 2. Crie e ative o ambiente virtual
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Configure as variáveis de ambiente
cp .env.example .env
# Edite .env com sua chave da API Groq e NODE3_MOCK_MODE=false
```

Obtenha sua chave gratuita em: https://console.groq.com/

---

## Configuração (.env)

```env
GROQ_API_KEY=sua_chave_aqui
NODE3_MOCK_MODE=false        # false = IA real | true = extração por regex (dev)
DEFAULT_MODEL=qwen/qwen3.8-27b
CHROMA_PERSIST_DIRECTORY=./data/vector_db
CHROMA_COLLECTION_NAME=licitacoes
```

---

## Uso

### Interface web (recomendado)

```bash
python start_app.py
```

Aguarde as mensagens de inicialização (~25s no primeiro start por conta do carregamento dos modelos).

- Streamlit: http://localhost:8501
- API Swagger: http://127.0.0.1:8000/docs

Ou inicie manualmente em dois terminais:

```bash
# Terminal 1
python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000

# Terminal 2
streamlit run app/streamlit_app.py
```

### Programático

```python
from dotenv import load_dotenv
load_dotenv()

from app.rag.langgraph_workflow import run_licit_graph_pipeline

result = run_licit_graph_pipeline("data/raw/uploads/edital.pdf")

# Respostas RAG por pergunta
for key, answer in result["analysis"]["rag_answers"].items():
    print(f"\n{answer['label']}")
    print(answer["resposta"])

# Checklist de documentos
print(result["checklist"]["resumo"])
```

---

## Estrutura do projeto

```
LicitGraphAi/
├── app/
│   ├── api/
│   │   └── main.py               # FastAPI: endpoints /upload, /analyze, /status
│   ├── rag/
│   │   ├── pdf_reader.py         # Extração e limpeza de texto do PDF
│   │   ├── text_chunker.py       # Fragmentação por parágrafos/seções
│   │   ├── node_1_reader_chunker.py
│   │   ├── node_2_embeddings.py  # Embeddings + ChromaDB
│   │   ├── node_3_analyzer.py    # RAG + LLM (Groq)
│   │   ├── node_4_document_generator.py  # Checklist categorizado
│   │   └── langgraph_workflow.py # Orquestração completa
│   └── streamlit_app.py          # Interface web
├── data/
│   ├── raw/uploads/              # PDFs enviados (ignorado pelo git)
│   ├── processed/                # Dados processados (ignorado pelo git)
│   └── vector_db/                # ChromaDB (ignorado pelo git)
├── tests/                        # Testes automatizados
├── examples/                     # Scripts de exemplo
├── docs/
├── start_app.py                  # Inicia FastAPI + Streamlit
├── .env.example
└── requirements.txt
```

---

## Testes

```bash
pytest tests/ -v
```

---

## Desenvolvimento

```bash
black app/
ruff check app/
```
