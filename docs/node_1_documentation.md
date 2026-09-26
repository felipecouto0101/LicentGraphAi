# Nó 1: Leitor e Fragmentador (RAG com LangChain)

## Descrição

O Nó 1 é responsável por extrair o texto de arquivos PDF de editais de licitação e fragmentá-lo em partes menores e mais gerenciáveis. Este é o primeiro nó do fluxo do LicitGraphAi e prepara os dados para os nós subsequentes.

## Funcionalidades

### 1. Extração de Texto PDF
- Utiliza `pdfplumber` para extração robusta de texto
- Suporta PDFs complexos e multi-páginas
- Validação de integridade do arquivo
- Contagem de páginas e caracteres

### 2. Fragmentação Inteligente
- Usa `RecursiveCharacterTextSplitter` do LangChain
- Configuração flexível de tamanho e sobreposição de chunks
- Separação inteligente por seções típicas de editais:
  - **Bens/Serviços**: Objeto da licitação, itens, especificações
  - **Prazos**: Prazos de entrega, vigência, cronogramas
  - **Exigências Técnicas**: Requisitos técnicos, normas, especificações
  - **Documentação**: Habilitação, certidões, licenças
  - **Outros**: Informações gerais

### 3. Metadados
- Geração automática de metadados para cada chunk
- Contagem de caracteres, palavras e linhas
- Identificação de seção e posição no documento
- Suporte a metadados personalizados

## Estrutura do Código

### Componentes Principais

#### `PDFReader` (`app/rag/pdf_reader.py`)
- `load_pdf(pdf_path)`: Extrai texto do PDF
- `get_text()`: Retorna o texto extraído
- `get_page_count()`: Retorna número de páginas
- `validate_pdf()`: Valida integridade do PDF

#### `TextChunker` (`app/rag/text_chunker.py`)
- `chunk_text(text)`: Divide texto em chunks padrão
- `chunk_by_sections(text)`: Divide por seções de editais
- `get_chunk_metadata()`: Gera metadados do chunk
- `create_document_chunks()`: Cria chunks no formato LangChain

#### `Node1ReaderChunker` (`app/rag/node_1_reader_chunker.py`)
- `process_pdf(pdf_path, metadata)`: Processamento completo
- `validate_output(output)`: Validação do resultado
- `get_section_summary()`: Resumo por seção

## Uso

### Uso Básico

```python
from app.rag.node_1_reader_chunker import process_edital_pdf

result = process_edital_pdf(
    pdf_path="data/raw/edital.pdf",
    chunk_size=1000,
    chunk_overlap=200,
    chunk_by_sections=True
)

print(f"Chunks gerados: {result['total_chunks']}")
print(f"Páginas: {result['page_count']}")
```

### Uso Avançado

```python
from app.rag.node_1_reader_chunker import Node1ReaderChunker

node = Node1ReaderChunker(
    chunk_size=500,
    chunk_overlap=100,
    chunk_by_sections=True
)

custom_metadata = {
    "edital_number": "123/2024",
    "orgao": "Prefeitura Municipal"
}

result = node.process_pdf("edital.pdf", metadata=custom_metadata)
```

### Processamento de Texto Direto

```python
from app.rag.text_chunker import TextChunker

chunker = TextChunker(chunk_size=200, chunk_overlap=50)
sections = chunker.chunk_by_sections(texto_do_edital)

for section, chunks in sections.items():
    print(f"{section}: {len(chunks)} chunks")
```

## Parâmetros de Configuração

### PDFReader
- Sem parâmetros obrigatórios

### TextChunker
- `chunk_size` (int): Tamanho máximo do chunk em caracteres (padrão: 1000)
- `chunk_overlap` (int): Sobreposição entre chunks (padrão: 200)
- `separators` (list): Separadores personalizados para divisão

### Node1ReaderChunker
- `chunk_size` (int): Tamanho dos chunks (padrão: 1000)
- `chunk_overlap` (int): Sobreposição (padrão: 200)
- `chunk_by_sections` (bool): Organiza por seções (padrão: True)

## Formato de Saída

```python
{
    "full_text": "texto completo do edital",
    "page_count": 15,
    "chunks": [
        {
            "content": "texto do chunk",
            "section": "bens",
            "metadata": {
                "chunk_id": 0,
                "total_chunks": 25,
                "char_count": 950,
                "word_count": 150,
                "line_count": 12
            }
        }
    ],
    "chunks_by_section": {
        "bens": ["chunk1", "chunk2"],
        "prazos": ["chunk3"],
        "exigencias_tecnicas": ["chunk4"],
        "documentacao": ["chunk5"],
        "outros": ["chunk6"]
    },
    "total_chunks": 25,
    "processing_method": "by_sections",
    "metadata": {}  # metadados personalizados
}
```

## Integração com LangGraph

Este nó será integrado ao fluxo do LangGraph como o primeiro nó:

```python
from langgraph.graph import StateGraph
from app.rag.node_1_reader_chunker import Node1ReaderChunker

def node_1_function(state):
    """Função do Nó 1 no LangGraph."""
    node = Node1ReaderChunker()
    result = node.process_pdf(state["pdf_path"])
    
    state["text_data"] = result
    return state
```

## Testes

Execute os testes com:

```bash
pytest tests/test_node_1.py -v
```

## Exemplos

Exemplos completos de uso estão disponíveis em:

- `examples/example_node_1.py`

## Dependências

- `pdfplumber>=0.10.0` - Extração de PDF
- `langchain>=0.1.0` - Framework de chunking
- `PyPDF2>=3.0.0` - Backup para PDF

## Próximos Passos

Após o Nó 1, o fluxo continua com:

1. **Nó 2**: Geração de Embeddings e armazenamento no ChromaDB
2. **Nó 3**: Análise de requisitos do edital
3. **Nó 4**: Comparação com perfil da empresa
4. **Nó 5**: Geração de relatório de riscos

## Considerações

- O tamanho ideal de chunk depende do modelo de LLM usado
- Sobreposição ajuda a manter contexto entre chunks
- Separação por seções melhora a precisão da análise
- Metadados são cruciais para rastreamento e debugging
