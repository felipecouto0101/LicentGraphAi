# LicitGraphAi

Organize assuntos de editais em PDF em **temas e subtemas com nomes claros**. A quantidade e os agrupamentos são definidos pela IA conforme o documento, sem cinco categorias fixas.

A etapa atual entrega somente o mapa de assuntos. Chat, explicações e o antigo botão de renomear seções foram retirados da interface e dos endpoints do mapa para serem trabalhados em uma etapa posterior.

## Como usar

1. Envie um PDF com texto extraível pela lateral da interface e clique em **Organizar assuntos**.
2. Acompanhe a leitura e os lotes de organização. A IA lê lotes de trechos completos para identificar subtemas, sem gerar explicações.
3. Busque um assunto ou abra um tema. Os subtemas mostram páginas físicas, citações verificadas dos trechos vinculados e suas seções de origem. Referências a anexos sem cabeçalho identificado aparecem em uma área separada.
4. Se a organização falhar depois de concluir lotes, os subtemas validados aparecem em um **mapa parcial**, claramente marcado como incompleto. O índice original é exibido quando ainda não existem subtemas validados; marcadores “Visão geral da seção” não são apresentados como subtemas. **Retomar organização com IA** reutiliza lotes válidos na mesma sessão, sem reenviar o PDF.

## Instalação

Requer Python 3.11 ou 3.12. Na raiz do projeto:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Configure no `.env` para usar Gemini no mapa de assuntos:

```dotenv
TOPIC_LLM_PROVIDER=gemini
GEMINI_API_KEY=sua_chave_local
GEMINI_MODEL=gemini-3.5-flash-lite
GEMINI_RPM_BUDGET=10
GEMINI_TPM_BUDGET=100000
GEMINI_RPD_BUDGET=500
GEMINI_OUTPUT_TOKEN_BUDGET=8192
NODE3_MOCK_MODE=false
```

Não envie o `.env` ao GitHub. A organização usa IA real. Inicie:

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
3. O texto completo é apresentado em linhas curtas com números explícitos (até 160 caracteres). Os IDs da chamada são curtos (p1, p2 etc.), vinculados às fontes originais no backend. A IA identifica subtemas e seleciona IDs dos trechos e de uma a três linhas por fonte. O backend verifica os índices e monta citações diretamente do texto original, sem pedir à IA para copiá-las. Todos os IDs do lote precisam estar cobertos antes de salvá-lo. Índices numéricos escritos como strings são normalizados e repetições removidas, sem ajustar índices fora do intervalo. Uma referência inválida recebe uma tentativa de correção; depois o lote é reduzido, sem aceitar fontes inventadas.
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
| Gemini Flash-Lite | Organização do mapa via REST, JSON estruturado e orçamento de saída próprio. |
| Groq e Qwen | Provedor alternativo do mapa e modelo do fluxo completo anterior. |
| FastAPI | Upload, execução em background, progresso e retomada. |
| Streamlit | Envio do documento, temas expansíveis, busca, fontes e mensagens de espera e recuperação. |
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

Cobertura de IDs confirma que todos os trechos enviados receberam ao menos uma referência, mas não garante que a IA identificou cada assunto dentro deles. Citação literal confirma a origem do trecho, não a adequação semântica de todo título. Consolidações limitadas a lotes podem deixar sinônimos em grupos diferentes. Os nomes e agrupamentos devem ser avaliados com editais variados. PDFs digitalizados precisam de OCR, ainda não implementado neste fluxo. Na interface, resumo do documento, status, navegação e anexos têm áreas delimitadas. O índice de contingência usa rótulos de seções e mantém o erro visível, evitando parecer um mapa final da IA. Subtemas válidos permanecem visíveis após interrupção, mesmo quando a consolidação final não terminou.

## Verificação

```powershell
python -m unittest discover -s tests -p 'test_topic*.py' -v
python -m py_compile app/rag/topic_map.py app/rag/topic_organizer.py app/api/main.py app/streamlit_app.py
```

Testes simulam respostas da IA e cobrem texto completo, páginas específicas por subtema, referências de linha inválidas, omissões, consolidação de sinônimos, divisão após truncamento, reaproveitamento e invalidação de cache, referências a anexos e fluxo da API. Execução com Groq e avaliação visual do Streamlit precisam das dependências completas e chave configuradas.

### Contas Groq independentes

Para usar três chaves de organizações distintas, configure no `.env`:

```dotenv
GROQ_API_KEY=sua_primeira_chave
GROQ_API_KEY_2=sua_segunda_chave
GROQ_API_KEY_3=sua_terceira_chave
GROQ_INDEPENDENT_ACCOUNTS=true
```

Reinicie o backend após alterar a configuração. O cliente mantém a conta atual enquanto disponível; quando ela entra em espera por RPM, tokens ou 429, procura outra disponível antes de aguardar. Cotas, janelas de requisições e teto de saída são separados por chave dentro de cada cliente. Chaves repetidas são descartadas. Os logs identificam a conta pelo número, sem exibir credenciais.

O padrão de `GROQ_INDEPENDENT_ACCOUNTS` é `false`: chaves da mesma organização compartilham os limites da Groq e não devem ativar essa opção. O controle é local ao cliente e não coordena processos ou aplicações externos usando as mesmas contas. Todas as contas indisponíveis, falhas persistentes ou pedidos grandes demais continuam interrompendo com erro recuperável. Trocar contas não corrige citações inválidas nem respostas truncadas. Não há garantia de aceleração de três vezes.

Testes do escalonador (sem chamadas à API): `python -m pytest tests/test_groq_account_scheduler.py -q`.

### Gemini no mapa de assuntos

`TOPIC_LLM_PROVIDER=gemini` seleciona o Gemini somente para o mapa; `/analyze/full` continua usando o pipeline Groq/LangGraph anterior. Sem seleção explícita, uma `GEMINI_API_KEY` configurada seleciona Gemini; caso contrário, usa Groq. Para voltar, configure `TOPIC_LLM_PROVIDER=groq` e reinicie o backend.

O cliente REST usa a biblioteca padrão do Python, sem uma nova dependência. Solicita JSON com esquema para identificação e agrupamento, mantém validação das linhas originais e rejeita respostas truncadas. Para Gemini, a identificação começa com até 12 trechos ou 16.000 caracteres por lote, saída de até 8.192 tokens e pensamento mínimo no modelo indicado. Lotes que ainda não couberem são subdivididos sem publicar respostas incompletas.

Os valores de 10 RPM, 100.000 tokens de entrada/minuto e 500 RPD são orçamentos locais baseados nas cotas informadas para este projeto; não são cotas universais nem saldo consultado no Google. O cliente espaça inícios de chamadas em cerca de 6,1 segundos, estima a entrada conservadoramente e atualiza o consumo com `usageMetadata`. Não impõe a espera de um minuto nem o teto de saída da Groq ao Gemini. Esperas, novas tentativas, correções e subdivisões são publicadas no progresso da interface, com contagem regressiva de espera.

Reservas locais são compartilhadas entre jobs com a mesma chave/modelo no mesmo processo. Para várias chaves do mesmo projeto, defina o mesmo `GEMINI_QUOTA_GROUP` em todos os clientes. O contador diário usa uma janela móvel conservadora de 24 horas e fica em memória. Ele não representa o saldo real do projeto, não acompanha outros processos/aplicações e é zerado ao reiniciar; a API continua sendo a autoridade sobre as cotas. Um 429 respeita o prazo retornado; falhas temporárias têm até três novas tentativas. Interrupções preservam apenas lotes já validados na sessão, conforme a seção de retomada.

Validação sem credenciais:

```bash
python -m unittest discover -s tests -p 'test_gemini*.py' -v
python -m unittest discover -s tests -p 'test_topic*.py' -v
```

Os testes simulam REST, cotas compartilhadas, RPM/TPM/RPD, recuperação de 429/503, autenticação, truncamento, integração upload → mapa e mensagens do frontend. Velocidade e qualidade com a API real precisam ser medidas com a chave local e o mesmo edital; não há garantia de aceleração.

### Leitura dos trechos de origem

A interface agrupa linhas verificadas consecutivas da mesma fonte e página em um bloco, com indicação de página uma única vez. Quebras de frase do PDF são refluídas; parágrafos, alíneas e cláusulas numeradas mantêm separação. Linhas não selecionadas não são inseridas, e lacunas, páginas e fontes diferentes permanecem em blocos distintos. O texto é escapado antes da renderização, sem executar HTML recebido do documento. A área “Seções vinculadas no PDF” foi removida; páginas e trechos continuam acessíveis em “Conferir trechos de origem”.

Novas extrações guardam os separadores originais entre fragmentos de linha. Mapas antigos com índices de linha também se agrupam, mas sem essa informação não é possível reconstruir todas as quebras de parágrafo originais. O cache de identificação foi versionado para não reutilizar resultados sem os novos separadores em uma retomada. A apresentação não faz chamadas adicionais à IA nem gera explicações.

Teste: `python -m unittest discover -s tests -p 'test_source_display.py' -v`.

### Contexto original e sobreposição de chunks

O leitor agora mantém as páginas completas na sessão do mapa. Depois da identificação, o backend localiza as citações nessas páginas, sem uma nova chamada à IA, e recupera o item numerado ou parágrafo que as contém. Trechos de chunks diferentes que pertencem ao mesmo intervalo original são exibidos uma única vez; a deduplicação não usa semelhança de frases. Os índices e evidências da IA permanecem armazenados, e a apresentação indica quando mostra contexto recuperado do documento.

Itens como 11.5 e 11.5.1 são separados, e a recuperação termina no próximo cabeçalho/item identificado. Alíneas pertencentes ao item permanecem juntas. Quando a citação aparece mais de uma vez, o chunk original só desambigua se tiver uma localização única. Se página, origem ou limites não puderem ser confirmados, mantém-se o trecho selecionado e um aviso; páginas distintas nunca são fundidas. Um item possivelmente interrompido no fim da página é sinalizado, sem inventar uma continuação. Blocos excepcionalmente grandes têm recuperação limitada. Isso é recuperação de contexto original, não uma explicação nem uma garantia de segmentação perfeita para qualquer diagramação.

Atualize e reinicie a aplicação, depois envie o PDF novamente para guardar as páginas completas na sessão. Mapas de sessões antigas sem esse texto continuam com os trechos selecionados. Testes: `python -m unittest discover -s tests -p 'test_source*.py' -v`.

### Tabelas nas fontes do PDF

O leitor preserva tabelas detectadas por geometria no PDF, com linhas, colunas e células mescladas. Na consulta às fontes, essas regiões são exibidas como tabelas roláveis, em vez de serem refluídas como um parágrafo. Uma continuação em outra página permanece separada e mantém a página original. Cabeçalhos/valores não são completados nem inferidos quando estiverem ausentes.

Códigos numéricos seguidos de hífen, como códigos de cargos, não são tratados como cláusulas. Regiões de tabela confirmadas também ficam fora da identificação de limites de itens numerados, e uma cláusula anterior termina antes da região tabular. Seleções repetidas são ocultadas quando todas as suas ocorrências já estão cobertas por intervalos recuperados; seleções ambíguas ainda não cobertas continuam identificadas como tais, sem duplicar o mesmo texto na mesma página.

Foi conferido o PDF UFBA: páginas 13–15 têm nove regiões tabulares detectadas, incluindo a continuação da tabela 10.7. A extração usa pdfplumber já instalado e não exige chamada à IA. Tabelas sem geometria detectável, imagens e PDFs com diagramação irregular ainda podem exigir conferência do PDF original; isso não implementa OCR nem garante extração perfeita. Reinicie e reenvie o documento para incluir os novos metadados de tabelas.

Teste: `python -m unittest discover -s tests -p 'test_pdf_tables.py' -v`.

### Falsas tabelas e continuações entre páginas

A presença de uma grade detectada não basta para apresentar uma tabela: o leitor rejeita blocos de uma coluna e layouts compostos por números de cláusulas e prosa longa. Isso evita transformar alinhamentos de parágrafos em tabelas sem sentido. O filtro é conservador e pode exigir conferência em documentos com layouts incomuns.

Itens numerados que terminam sem fechar a frase podem recuperar sua continuação no começo da página seguinte quando há sinais compatíveis, sem cabeçalho ou tabela intermediária. O bloco mantém o número do item e as duas páginas, remove rodapés identificados e termina antes do próximo item. Não corrige maiúsculas nem reescreve o PDF: restaura o começo da frase quando consegue confirmar essa ligação. Quando os sinais não são suficientes, mantém os trechos separados e o aviso.

Tabelas em páginas adjacentes podem ser vinculadas se as grades estiverem nas bordas das páginas, as colunas e larguras forem compatíveis e não houver novo título de tabela. As partes continuam em blocos por página, sem inventar valores em células vazias. Selecionar uma das partes também recupera as demais partes vinculadas. No PDF UFBA, a tabela 2.1 tem os cargos 201–205 na primeira página e continua com 206, 207 e cargos de nível superior na segunda página. O item 22.12 começa na página 33 e termina na 34; as falsas tabelas dessas páginas foram rejeitadas em conferência local.
