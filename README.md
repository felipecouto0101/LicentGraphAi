# LicitGraphAi

Organize assuntos de editais em PDF em **temas e subtemas com nomes claros**. A quantidade e os agrupamentos são definidos pela IA conforme o documento, sem cinco categorias fixas.

A etapa atual entrega somente o mapa de assuntos. Chat, explicações e o antigo botão de renomear seções foram retirados da interface e dos endpoints do mapa para serem trabalhados em uma etapa posterior.

## Como usar

1. Envie um PDF com texto extraível pela lateral da interface e clique em **Organizar assuntos**.
2. Acompanhe a leitura e os lotes de organização. A IA lê lotes de trechos completos para identificar subtemas, sem gerar explicações.
3. Busque um assunto ou abra um tema. Os subtemas mostram páginas físicas, citações verificadas dos trechos vinculados e suas seções de origem. Referências a anexos sem cabeçalho identificado aparecem em uma área separada.
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
2. Todos os chunks são enviados com seu texto completo, IDs e páginas. Chunks acima de 2.400 caracteres são fragmentados sem descarte; lotes são limitados a oito entradas e 8.000 caracteres de texto.
3. O texto completo é apresentado em linhas curtas numeradas (até 160 caracteres). A IA identifica subtemas e seleciona IDs dos trechos e de uma a três linhas por fonte. O backend verifica os índices e monta citações diretamente do texto original, sem pedir à IA para copiá-las. Todos os IDs do lote precisam estar cobertos antes de salvá-lo. Uma referência inválida recebe uma tentativa de correção; depois o lote é reduzido, sem aceitar fontes inventadas.
4. Nomes e sugestões de tema são consolidados em lotes compactos de até 12 assuntos. O modelo pode reunir sinônimos; duplicações de mesmo nome no grupo final são reunidas. Todos os assuntos precisam ser preservados, sem IDs inventados ou repetidos.
5. Cada subtema recebe somente as páginas, chunks e citações vinculados a ele; não herda todas as páginas da seção estrutural. A interface permite abrir os trechos utilizados.
6. Uma conferência local compara menções a anexos com cabeçalhos identificados no arquivo, inclusive equivalência entre números romanos e arábicos. Uma referência não localizada não prova ausência: o anexo pode estar em outro arquivo ou ter formatação não reconhecida.

O número de chamadas varia com o volume de texto, assuntos identificados e consolidações. Ler todos os trechos aumenta consumo e tempo em relação às três amostras da versão anterior. Sob o limite observado de uma chamada por minuto, a organização pode levar dezenas de minutos; a saída curta não elimina a espera imposta pela cota. Divisões por tamanho/truncamento podem acrescentar chamadas. Não são geradas explicações individuais.

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
- `GET /job/{job_id}`: informa progresso e devolve `result.topic_map`, `map_version=4` e `organization_status`.
- `POST /job/{job_id}/organize-map`: retoma uma organização interrompida. Recusa execução concorrente do mesmo job.
- `POST /analyze/full`: mantém o pipeline LangGraph anterior, fora desta interface.

Jobs, documentos e lotes validados da organização ficam **em memória**. Uma falha de API permite retomar na mesma sessão; reiniciar o backend exige novo envio. O cache usa versão e hash do conteúdo enviado, evitando reutilizar lotes de texto diferente. O mapa não utiliza os checkpoints persistentes da análise completa. Os arquivos antigos em `data/checkpoints/` permanecem disponíveis ao fluxo anterior.

## Cotas e limites

A Groq informou um teto de 1.000 tokens de saída por minuto no cenário observado. O cliente usa `GROQ_OUTPUT_TOKEN_BUDGET=950` por padrão e pode aguardar aproximadamente um minuto entre chamadas. A organização ainda pode levar minutos, dependendo de quantidade de seções, cotas e falhas. Não há garantia de tempo.

| Variável | Uso |
| --- | --- |
| `GROQ_API_KEY` | Chave para organização com IA. |
| `NODE3_MOCK_MODE` | Deixe `false` para usar IA real. |
| `GROQ_OUTPUT_TOKEN_BUDGET` | Teto preventivo de tokens de saída. |
| `GROQ_RPM_BUDGET` | Limite local de chamadas por minuto. |
| `CHROMA_PERSIST_DIRECTORY` | Diretório vetorial do fluxo completo anterior. |

Cobertura de IDs confirma que todos os trechos enviados receberam ao menos uma referência, mas não garante que a IA identificou cada assunto dentro deles. Citação literal confirma a origem do trecho, não a adequação semântica de todo título. Consolidações limitadas a lotes podem deixar sinônimos em grupos diferentes. Os nomes e agrupamentos devem ser avaliados com editais variados. PDFs digitalizados precisam de OCR, ainda não implementado neste fluxo. Na interface, resumo do documento, status, navegação e anexos têm áreas delimitadas. O índice de contingência usa rótulos de seções/subseções e mantém o erro visível, evitando parecer um mapa final da IA.

## Verificação

```powershell
python -m unittest discover -s tests -p 'test_topic*.py' -v
python -m py_compile app/rag/topic_map.py app/rag/topic_organizer.py app/api/main.py app/streamlit_app.py
```

Testes simulam respostas da IA e cobrem texto completo, páginas específicas por subtema, referências de linha inválidas, omissões, consolidação de sinônimos, divisão após truncamento, reaproveitamento e invalidação de cache, referências a anexos e fluxo da API. Execução com Groq e avaliação visual do Streamlit precisam das dependências completas e chave configuradas.
