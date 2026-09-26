# LicitGraphAi - Analisador e Auditor Autônomo de Editais de Licitação

Sistema inteligente para análise automática de editais de licitação pública (Nova Lei 14.133), utilizando IA para extrair pontos críticos, avaliar aptidão da empresa e gerar relatórios de riscos.

## Stack Tecnológica

### Core
- **Python 3.11+** - Linguagem principal
- **LangGraph** - Orquestração do fluxo do agente
- **LangChain** - Componentes de IA & RAG

### Inteligência Artificial
- **Groq + Llama 3.1** - Modelo de linguagem 100% gratuito
- **ChromaDB** - Banco de dados vetorial para RAG

### Backend
- **FastAPI** - Framework web moderno e rápido
- **Pydantic** - Validação de dados

### Processamento de Documentos
- **pdfplumber** - Extração robusta de texto de PDFs
- **PyPDF2** - Backup/complemento para PDF

### Frontend
- **Streamlit** - Interface em Python

### Banco de Dados
- **SQLite** - Metadados (desenvolvimento)
- **PostgreSQL** - Metadados (produção)

### Utilitários
- **python-dotenv** - Variáveis de ambiente
- **pytest** - Testes
- **black/ruff** - Formatação e linting

## Arquitetura do Sistema

### Fluxo de Nós (LangGraph)

1. **Nó 1: Leitor e Fragmentador** ✅
   - Extrai texto do PDF do edital
   - Fragmenta em partes menores (bens, prazos, exigências técnicas)
   - Usa LangChain para chunking inteligente

2. **Nó 2: Geração de Embeddings** (Próximo)
   - Gera embeddings dos chunks
   - Armazena no ChromaDB

3. **Nó 3: Análise de Requisitos** (Pendente)
   - Analisa requisitos do edital
   - Identifica pontos críticos

4. **Nó 4: Comparação com Perfil** (Pendente)
   - Compara requisitos com perfil da empresa
   - Avalia aptidão

5. **Nó 5: Relatório de Riscos** (Pendente)
   - Gera relatório final
   - Classifica riscos

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

4. Configure as variáveis de ambiente:
```bash
cp .env.example .env
# Edite .env com sua chave da API Groq
```

5. Obtenha sua chave da API Groq em: https://console.groq.com/

## Estrutura do Projeto

```
LicitGraphAi/
├── app/
│   ├── agents/          # Agentes LangGraph
│   ├── rag/             # Componentes RAG
│   │   ├── pdf_reader.py
│   │   ├── text_chunker.py
│   │   └── node_1_reader_chunker.py
│   ├── api/             # API FastAPI
│   ├── models/          # Modelos de dados
│   └── utils/           # Utilitários
├── data/
│   ├── raw/             # PDFs originais
│   ├── processed/       # Dados processados
│   └── vector_db/       # ChromaDB
├── tests/               # Testes
├── docs/                # Documentação
├── examples/            # Exemplos de uso
└── requirements.txt
```

## Uso

### Exemplo Básico

```python
from app.rag.node_1_reader_chunker import process_edital_pdf

result = process_edital_pdf("data/raw/edital.pdf")
print(f"Chunks gerados: {result['total_chunks']}")
```

### Executar Exemplos

```bash
python examples/example_node_1.py
```

### Executar Testes

```bash
pytest tests/test_node_1.py -v
```

## Desenvolvimento

### Formatação de Código
```bash
black app/
ruff check app/
```

### Testes
```bash
pytest tests/ -v
```

## Documentação

- [Documentação do Nó 1](docs/node_1_documentation.md)
- [Exemplos](examples/)

## Contribuição

Este projeto está em desenvolvimento. Contribuições são bem-vindas!

## Licença

[Adicionar licença aqui]

## Contato

[Adicionar informações de contato]
