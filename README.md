# LicitGraphAi - Analisador e Auditor Autônomo de Editais de Licitação

Sistema inteligente para análise automática de editais de licitação pública (Nova Lei 14.133), utilizando IA para extrair pontos críticos, avaliar aptidão da empresa e gerar relatórios de riscos.

## Stack Tecnológica (Implementado)

### Core
- **Python 3.11+** - Linguagem principal
- **LangChain** - Componentes de RAG (RecursiveCharacterTextSplitter)

### Processamento de Documentos
- **pdfplumber** - Extração robusta de texto de PDFs
- **PyPDF2** - Backup/complemento para PDF

### Inteligência Artificial
- **sentence-transformers** - Modelo de embeddings gratuito (all-MiniLM-L6-v2)
- **ChromaDB** - Banco de dados vetorial para RAG

### Utilitários
- **python-dotenv** - Variáveis de ambiente
- **pytest** - Testes
- **black/ruff** - Formatação e linting

## Arquitetura do Sistema

### Fluxo de Nós Implementados

1. **Nó 1: Leitor e Fragmentador** ✅
   - Extrai texto do PDF do edital
   - Fragmenta em partes menores (bens, prazos, exigências técnicas)
   - Usa LangChain para chunking inteligente
   - Suporta PDFs bem estruturados e mal estruturados (fallback)

2. **Nó 2: Geração de Embeddings** ✅
   - Gera embeddings dos chunks usando sentence-transformers
   - Armazena no ChromaDB (banco vetorial)
   - Suporta busca semântica
   - Processamento em batch para performance

### Próximos Nós (Pendentes)
- Nó 3: Análise de Requisitos (Groq + Llama 3.1)
- Nó 4: Comparação com Perfil da Empresa
- Nó 5: Geração de Relatório de Riscos

## Instalação

### Pré-requisitos
- Python 3.11 ou superior
- pip

### Setup

1. Clone o repositório
2. Crie um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Estrutura do Projeto

```
LicitGraphAi/
├── app/
│   └── rag/             # Componentes RAG implementados
│       ├── pdf_reader.py
│       ├── text_chunker.py
│       ├── node_1_reader_chunker.py
│       └── node_2_embeddings.py
├── data/
│   ├── raw/             # PDFs originais
│   ├── processed/       # Dados processados
│   └── vector_db/       # ChromaDB
├── tests/               # Testes (28 testes implementados)
├── docs/                # Documentação
├── examples/            # Exemplos de uso
└── requirements.txt
```

## Uso

### Exemplo Básico (Nó 1)

```python
from app.rag.node_1_reader_chunker import process_edital_pdf

result = process_edital_pdf("data/raw/edital.pdf")
print(f"Chunks gerados: {result['total_chunks']}")
```

### Exemplo de Integração (Nó 1 + Nó 2)

```python
from app.rag.node_2_embeddings import Node2EmbeddingGenerator

node2 = Node2EmbeddingGenerator()
result = node2.process_chunks(chunks, persist_directory="./data/vector_db")
print(f"Embeddings gerados: {result['total_embeddings']}")
```

### Executar Exemplos

```bash
python examples/example_node_1.py
python examples/example_node_1_2_integration.py
```

### Executar Testes

```bash
pytest tests/test_node_1.py -v  # Testes do Nó 1
pytest tests/test_node_2.py -v  # Testes do Nó 2
pytest tests/ -v                # Todos os testes
```

## Desenvolvimento

### Testes
```bash
pytest tests/ -v
```

### Formatação de Código
```bash
black app/
ruff check app/
```

## Documentação

- [Documentação do Nó 1](docs/node_1_documentation.md)
- [Exemplos de Integração](examples/example_node_1_2_integration.py)
