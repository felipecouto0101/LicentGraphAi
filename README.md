# LicitGraphAi - Analisador e Auditor Autônomo de Editais de Licitação

Sistema inteligente para análise automática de editais de licitação pública (Nova Lei 14.133), utilizando IA para extrair pontos críticos, avaliar aptidão da empresa e gerar relatórios de riscos.

## Stack Tecnológica (Implementado)

### Core
- **Python 3.11+** - Linguagem principal
- **LangChain** - Componentes de RAG (RecursiveCharacterTextSplitter)
- **LangGraph** - Orquestração de workflow com estado compartilhado

### Processamento de Documentos
- **pdfplumber** - Extração robusta de texto de PDFs
- **PyPDF2** - Backup/complemento para PDF

### Inteligência Artificial
- **Groq API** + **OpenAI GPT-OSS-120b** - Modelo de linguagem gratuito
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

3. **Nó 3: Análise de Requisitos com IA** ✅
   - Analisa requisitos usando Groq API + OpenAI GPT-OSS-120b
   - Extrai informações estruturadas (técnicos, prazos, documentação, objeto, riscos)
   - Identifica pontos críticos e classifica riscos (ALTO/MÉDIO/BAIXO)
   - Gera recomendações e resumo executivo
   - Integração completa com Nó 2

4. **Nó 4: Gerador de Checklist de Documentos** ✅
   - Extrai documentos necessários do edital
   - Categoriza documentos (habilitação, técnica, fiscal, jurídica, trabalhista)
   - Gera checklist estruturado com prazos e observações
   - Integração completa com Nó 3
   - Resumo e recomendações

### Orquestração com LangGraph
- **Workflow automático**: node_1 → node_2 → node_3 → node_4
- **Estado compartilhado**: Todos os nós acessam o mesmo estado
- **Orquestração centralizada**: Gerenciamento em um lugar só
- **Fácil escalar**: Adicionar novos nós é simples

### Próximos Nós (Pendentes)
- Nó 5: Comparação com Perfil da Empresa
- Nó 6: Geração de Relatório Final de Riscos

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
# Edite .env com sua chave da API Groq (opcional para desenvolvimento)
```

5. Obtenha sua chave da API Groq em: https://console.groq.com/ (opcional)

## Estrutura do Projeto

```
LicitGraphAi/
├── app/
│   └── rag/             # Componentes RAG implementados
│       ├── pdf_reader.py
│       ├── text_chunker.py
│       ├── node_1_reader_chunker.py
│       ├── node_2_embeddings.py
│       ├── node_3_analyzer.py
│       ├── node_4_document_generator.py
│       └── langgraph_workflow.py  # Orquestração LangGraph
├── data/
│   ├── raw/             # PDFs originais
│   ├── processed/       # Dados processados
│   └── vector_db/       # ChromaDB
├── tests/               # Testes (79 testes implementados)
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

### Exemplo de Análise com IA (Nó 3)

```python
from app.rag.node_3_analyzer import Node3RequirementAnalyzer

node3 = Node3RequirementAnalyzer(api_key="sua_chave_groq", mock_mode=False)
result = node3.process_from_node2(chunks)
print(f"Pontos críticos: {result['critical_analysis']}")
print(f"Nível de risco: {result['overall_risk_level']}")
```

### Exemplo de Geração de Checklist (Nó 4)

```python
from app.rag.node_4_document_generator import Node4DocumentGenerator

node4 = Node4DocumentGenerator(mock_mode=True)
result = node4.process_from_node3(node3_analysis)
print(f"Total de documentos: {result['resumo']['total_documentos']}")
print(f"Categorias: {list(result['checklist'].keys())}")
```

### Exemplo com LangGraph (Workflow Completo)

```python
from app.rag.langgraph_workflow import run_licit_graph_pipeline

# Executa o pipeline completo automaticamente
final_state = run_licit_graph_pipeline("data/raw/edital.pdf")

# Acessa os resultados
print(f"Chunks: {len(final_state['chunks'])}")
print(f"Embeddings: {final_state['embeddings']['total_embeddings']}")
print(f"Análise: {final_state['analysis']}")
print(f"Checklist: {final_state['checklist']}")
```

### Executar Exemplos

```bash
python examples/example_node_1.py
python examples/example_node_1_2_integration.py
python examples/example_node_3_env.py
python examples/test_groq_api.py
python examples/example_langgraph.py  # Workflow LangGraph
```

### Executar Testes

```bash
pytest tests/test_node_1.py -v  # Testes do Nó 1
pytest tests/test_node_2.py -v  # Testes do Nó 2
pytest tests/test_node_3_step1.py -v  # Testes do Nó 3 (configuração)
pytest tests/test_node_3_step2.py -v  # Testes do Nó 3 (análise básica)
pytest tests/test_node_3_step3.py -v  # Testes do Nó 3 (extração estruturada)
pytest tests/test_node_3_step4.py -v  # Testes do Nó 3 (pontos críticos)
pytest tests/test_node_3_step5.py -v  # Testes do Nó 3 (integração)
pytest tests/test_node_4_step1.py -v  # Testes do Nó 4 (configuração)
pytest tests/test_node_4_step2.py -v  # Testes do Nó 4 (extração)
pytest tests/test_node_4_step3.py -v  # Testes do Nó 4 (categorização)
pytest tests/test_node_4_step4.py -v  # Testes do Nó 4 (checklist)
pytest tests/test_node_4_step5.py -v  # Testes do Nó 4 (integração)
pytest tests/test_langgraph.py -v  # Testes do LangGraph
pytest tests/ -v                # Todos os testes (79 testes)
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
- [Exemplos do Nó 3](examples/example_node_3_env.py)
- [Exemplos do LangGraph](examples/example_langgraph.py)
- [Teste de API Groq](examples/test_groq_api.py)
