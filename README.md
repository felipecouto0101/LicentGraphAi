# LicentGraphAi

Explore editais em PDF por meio de um **mapa de temas e subtemas**. O sistema lê as páginas, preserva suas referências, indexa os trechos e apresenta o mapa sem gerar centenas de explicações antecipadamente. Abra um assunto para pedir uma explicação ou faça uma pergunta livre ao edital.

## Como usar

1. Envie um PDF com texto extraível e clique em **Criar mapa de assuntos**.
2. Navegue pelos temas e subtemas ou procure pelo nome de um assunto. A página física do PDF aparece ao lado de cada subtema.
3. Clique em **Explicar este assunto** quando quiser uma interpretação em linguagem simples, com citações literais verificadas. A conversa também aceita perguntas livres.
4. Quando um cabeçalho original for pouco claro, clique em **Organizar nomes com IA** para sugerir nomes mais amigáveis dentro daquele tema. O título do PDF continua disponível e os IDs, trechos e páginas não mudam.

O mapa é um índice automático, não uma auditoria completa das exigências. Ele inclui todos os trechos de texto que o leitor conseguiu extrair do PDF, mas um título pode estar mal formatado, e PDFs digitalizados podem exigir OCR. A resposta do chat traz fontes para revisão; encontrar uma frase literal na página não prova, por si só, que toda interpretação da IA está correta.

## Instalação

Requer Python 3.11 ou 3.12. Na raiz do projeto, crie e ative um ambiente virtual e instale as dependências:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Para gerar explicações, configure `GROQ_API_KEY` no `.env` e mantenha `NODE3_MOCK_MODE=false`. O mapa inicial pode ser preparado sem uma chave; as perguntas e a organização opcional de nomes precisam da Groq real. Inicie:

```powershell
python start_app.py
```

Interface: **http://localhost:8501**. API: **http://127.0.0.1:8002/docs**. Em terminais separados:

```powershell
python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8002
python -m streamlit run app/streamlit_app.py
```

## Fluxo e tecnologias

| Tecnologia | Função |
| --- | --- |
| `pdfplumber` | Extrai o texto do PDF e associa trechos à página física. |
| Sentence Transformers (`all-MiniLM-L6-v2`) | Gera embeddings dos trechos e permite aproximar títulos de assuntos semelhantes. |
| ChromaDB | Armazena os embeddings e recupera trechos relacionados a perguntas livres. |
| Groq + Qwen | Explica um assunto solicitado, responde perguntas e opcionalmente sugere nomes mais claros. |
| LangChain | Fornece o cliente `ChatGroq` e as mensagens usadas nas chamadas à Groq. |
| FastAPI | Inicia a preparação em background e expõe o mapa e o chat. |
| Streamlit | Exibe o mapa, busca de assuntos, páginas e conversa. |
| LangGraph | Permanece no código do fluxo completo anterior (`app/rag/langgraph_workflow.py`); o fluxo rápido do mapa usa apenas leitura, indexação e consultas sob demanda. |

O índice inicial preserva as linhas do PDF, identifica seções antes de fragmentar o texto e mantém a sequência das páginas. Títulos quebrados em linhas são reunidos; entradas do sumário não criam novos assuntos. Cláusulas de anexos aparecem como subtemas. Títulos semanticamente próximos podem ser agrupados, com seus subtemas e fontes preservados. Títulos genéricos ou sem cabeçalho permanecem visíveis para evitar descartar texto. **A Groq não é chamada durante a criação do mapa.** Um clique para organizar nomes usa uma chamada apenas para os nomes daquele tema; a explicação usa uma chamada sobre trechos recuperados quando solicitada.

O fluxo LangGraph completo anterior permanece acessível pela API em `POST /analyze/full`, fora da interface do mapa. Ele mantém sua extração, checklist e checkpoints antigos.

O endpoint `POST /analyze/upload` retorna um `job_id`; `GET /job/{job_id}` informa progresso e devolve `result.topic_map`. `POST /job/{job_id}/ask` aceita `question`, `topic_id` opcional e um histórico curto. `POST /job/{job_id}/organize` aceita `theme_id`. Os jobs e os documentos preparados ficam em memória no processo da API; após reiniciar a API, envie o PDF novamente. A coleção vetorial persiste em `data/vector_db` e ainda não tem limpeza automática.

## Limites e custo de chamadas

A antiga extração em lotes e as centenas de explicações individuais **não são executadas pela interface do mapa**. Os arquivos antigos em `data/checkpoints/` não são apagados, mas o mapa rápido é reconstruído diretamente do PDF e não reaproveita as explicações antigas. Após atualizar a identificação de temas, reinicie a API e o Streamlit e envie novamente o PDF; mapas da versão anterior são recusados pela interface. A interface nova não apresenta o checklist de documentos do fluxo antigo; consulte o edital para uma conferência exaustiva.

A Groq informou um limite de 1.000 tokens de saída por minuto no cenário testado. O Nó 3 usa `GROQ_OUTPUT_TOKEN_BUDGET=950` por padrão e pode esperar aproximadamente um minuto entre perguntas seguidas. Agora esse custo só ocorre nos assuntos abertos pelo usuário. Outras cotas e a velocidade do modelo variam por conta e horário. A primeira indexação pode demorar pelo carregamento do modelo local; no PDF observado, a geração de 163 embeddings levou cerca de 14 segundos depois do carregamento. Isso não é uma garantia para outros PDFs ou computadores.

| Variável | Uso |
| --- | --- |
| `GROQ_API_KEY` | Chave para chat e organização opcional de nomes. |
| `NODE3_MOCK_MODE` | Deixe `false` para usar a IA real. |
| `GROQ_OUTPUT_TOKEN_BUDGET` | Máximo preventivo de tokens de saída do Nó 3. |
| `GROQ_RPM_BUDGET` | Teto local de chamadas por minuto. |
| `CHROMA_PERSIST_DIRECTORY` | Diretório da coleção vetorial. |

## Verificação

Execute `python -m unittest discover -s tests -p 'test_topic*.py'` para testar a cobertura dos trechos no mapa e o agrupamento sem perda de fontes. A compilação dos arquivos Python também pode ser verificada com `python -m py_compile app/rag/topic_map.py app/api/main.py app/streamlit_app.py`. O fluxo completo com PDF e Groq reais depende de testar na máquina que tem as dependências e a chave configuradas.
