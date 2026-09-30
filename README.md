# LicitGraphAi

Organize assuntos de editais em PDF em **temas e subtemas com nomes claros**. A quantidade e os agrupamentos são definidos pela IA conforme o documento, sem cinco categorias fixas.

A etapa atual entrega somente o mapa de assuntos. Chat, explicações e o antigo botão de renomear seções foram retirados da interface e dos endpoints do mapa para serem trabalhados em uma etapa posterior.

## Como usar

1. Envie um PDF com texto extraível pela lateral da interface e clique em **Organizar assuntos**.
2. Acompanhe a leitura e os lotes de organização. A IA recebe títulos e pequenas amostras do documento, sem gerar explicações.
3. Busque um assunto ou abra um tema. Os subtemas mostram páginas físicas do PDF e suas seções de origem.
4. Se a organização falhar, o índice estrutural continua disponível, claramente identificado. **Retomar organização com IA** reutiliza lotes válidos na mesma sessão, sem reenviar o PDF.

## Instalação

Requer Python 3.11 ou 3.12. Na raiz do projeto:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Configure `GROQ_API_KEY` no `.env` e mantenha `NODE3_MOCK_MODE=false`. A organização usa IA real. Inicie:

```powershell
python start_app.py
```

Interface: http://localhost:8501. API: http://127.0.0.1:8002/docs. Em terminais separados:

```powershell
python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8002
python -m streamlit run app/streamlit_app.py
```

Após atualizar para esta versão, reinicie os dois processos e envie novamente o PDF. Mapas da versão anterior precisam ser reconstruídos.

## Organização do mapa

1. `pdfplumber` extrai páginas preservando linhas. O leitor identifica seções antes de dividir em chunks, reúne títulos quebrados e evita interpretar o sumário como corpo.
2. Cada subtema estrutural recebe um ID e até três amostras de 240 caracteres, distribuídas entre início, meio e fim de seus trechos.
3. A IA organiza lotes de até seis entradas em temas e subtemas. Não há nomes nem quantidade de categorias predefinidos. Pedidos truncados ou grandes demais são divididos quando contêm mais de uma entrada.
4. Uma chamada compacta concilia temas dos lotes anteriores. Subtemas de mesmo nome dentro do grupo são reunidos, preservando todas as fontes.
5. A validação rejeita referências inventadas, seções omitidas e temas duplicados na consolidação. Os títulos originais, páginas e IDs dos chunks permanecem vinculados ao mapa final.

O número normal de chamadas é `ceil(subtemas estruturais / 6) + 1`. Para 48 entradas, são oito chamadas de organização e uma de consolidação; divisões, erros e novas tentativas podem aumentar esse total. Não são geradas centenas de explicações individuais.

O mapa atual **não gera embeddings nem indexa no ChromaDB**, pois consulta e chat estão fora desta etapa. Essas tecnologias continuam no pipeline completo anterior.

## Tecnologias

| Tecnologia | Papel |
| --- | --- |
| Python e pdfplumber | Leitura do PDF e preservação das páginas. |
| LangChain | Fragmentação do texto, mensagens e integração com a LLM. |
| Groq e Qwen | Organização de temas e subtemas; controle de cotas reutiliza o cliente existente. |
| FastAPI | Upload, execução em background, progresso e retomada. |
| Streamlit | Envio do documento, mapa em duas colunas, busca e fontes. |
| LangGraph | Orquestra o pipeline completo anterior, disponível em `/analyze/full`. |
| Sentence Transformers e ChromaDB | Embeddings e busca vetorial do pipeline completo anterior; não executados no mapa desta etapa. |

## API e retomada

- `POST /analyze/upload`: recebe PDF e inicia leitura e organização automática.
- `GET /job/{job_id}`: informa progresso e devolve `result.topic_map`, `map_version=3` e `organization_status`.
- `POST /job/{job_id}/organize-map`: retoma uma organização interrompida. Recusa execução concorrente do mesmo job.
- `POST /analyze/full`: mantém o pipeline LangGraph anterior, fora desta interface.

Jobs, documentos e lotes validados da organização ficam **em memória**. Uma falha de API permite retomar na mesma sessão; reiniciar o backend exige novo envio. O mapa não utiliza os checkpoints persistentes da análise completa. Os arquivos antigos em `data/checkpoints/` permanecem disponíveis ao fluxo anterior.

## Cotas e limites

A Groq informou um teto de 1.000 tokens de saída por minuto no cenário observado. O cliente usa `GROQ_OUTPUT_TOKEN_BUDGET=950` por padrão e pode aguardar aproximadamente um minuto entre chamadas. A organização ainda pode levar minutos, dependendo de quantidade de seções, cotas e falhas. Não há garantia de tempo.

| Variável | Uso |
| --- | --- |
| `GROQ_API_KEY` | Chave para organização com IA. |
| `NODE3_MOCK_MODE` | Deixe `false` para usar IA real. |
| `GROQ_OUTPUT_TOKEN_BUDGET` | Teto preventivo de tokens de saída. |
| `GROQ_RPM_BUDGET` | Limite local de chamadas por minuto. |
| `CHROMA_PERSIST_DIRECTORY` | Diretório vetorial do fluxo completo anterior. |

Cobertura de IDs garante que nenhuma seção indexada seja descartada, mas não comprova classificação semântica correta nem descoberta de todos os assuntos dentro de cada seção. Amostras podem não representar detalhes distantes. Os nomes e agrupamentos devem ser avaliados com editais variados. PDFs digitalizados precisam de OCR, ainda não implementado neste fluxo.

## Verificação

```powershell
python -m unittest discover -s tests -p 'test_topic*.py' -v
python -m py_compile app/rag/topic_map.py app/rag/topic_organizer.py app/api/main.py app/streamlit_app.py
```

Testes simulam respostas da IA e cobrem agrupamentos variáveis, referências e páginas, rejeição de omissões, consolidação, divisão após truncamento, reaproveitamento de lotes e fluxo da API. Execução com Groq e avaliação visual do Streamlit precisam das dependências completas e chave configuradas.
