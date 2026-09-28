# LicentGraphAi

Aplicação para organizar informações de editais em PDF. Lê o arquivo, extrai itens com a Groq, apresenta explicações em linguagem simples e separa documentos de participação, atividades da execução e anexos de consulta. **O resultado é uma análise automática para conferência, não substitui a leitura do edital.**

## Arquitetura e tecnologias

O processamento é um fluxo **linear de quatro nós** definido em `app/rag/langgraph_workflow.py`. O **LangGraph** usa `StateGraph` para executar os nós na ordem PDF → embeddings → análise → checklist e compartilhar entre eles o estado com os trechos, a referência à coleção vetorial, a análise, o checklist e eventuais erros. Neste projeto, o grafo não implementa agentes autônomos nem ramificações condicionais. A retomada descrita abaixo é implementada pelo nó 3 com arquivos JSON locais, e não pelo sistema de checkpoints do LangGraph.

| Tecnologia | Papel no projeto |
| --- | --- |
| Python | Linguagem da aplicação e do pipeline. |
| LangGraph | Orquestra os quatro nós e transmite o estado entre eles. |
| LangChain (`langchain-groq` e `langchain-core`) | Fornece `ChatGroq` e as mensagens `SystemMessage`/`HumanMessage` usadas nas chamadas aos modelos da Groq. A extração e a busca são implementadas nos nós do projeto. |
| Groq | Executa as chamadas de LLM para extração e redação; o modelo padrão nos nós 3 e 4 é `qwen/qwen3.8-27b`. |
| `pdfplumber` | Lê o texto do PDF, que depois é dividido em trechos com referência à página. |
| `sentence-transformers` | Gera embeddings dos trechos com `all-MiniLM-L6-v2` por padrão. |
| ChromaDB | Armazena os vetores e permite recuperar trechos semelhantes para a consulta RAG. |
| FastAPI e Uvicorn | Expõem a API de upload, consulta de jobs e status do serviço. |
| Streamlit | Exibe o formulário de upload, o progresso e as abas do resultado. |
| `python-dotenv` e pytest | Carregam a configuração `.env` e executam os testes, respectivamente. |

O **RAG** aqui combina a indexação dos trechos do edital no ChromaDB com a recuperação por similaridade durante a análise. O nó 3 também percorre os trechos em lotes para extrair informações; a recuperação vetorial não substitui essa leitura em lotes.

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

Abra **http://localhost:8501** para usar a interface. A documentação da API fica em **http://127.0.0.1:8002/docs**. Na primeira análise, o carregamento do modelo de embeddings pode demorar; a API não precisa carregar esse modelo para responder a `/status`.

Se preferir terminais separados, inicie a API e depois o Streamlit:

```powershell
python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8002
python -m streamlit run app/streamlit_app.py
```

No Linux ou macOS, ative o ambiente com `source .venv/bin/activate` e crie o `.env` com `cp .env.example .env`.

Se o iniciador indicar que a API encerrou ou não respondeu, execute a API sozinha para ver o erro original no terminal:

```powershell
python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8002 --log-level debug
```

Verifique também se o Python ativo é o do ambiente virtual (`python -c "import sys; print(sys.executable)"`). Corrija a causa mostrada no traceback antes de repetir `python start_app.py`.

## Configuração

| Variável | Uso |
| --- | --- |
| `GROQ_API_KEY` | Obrigatória para análise real. |
| `GROQ_API_KEY_2` a `GROQ_API_KEY_5` | Chaves adicionais opcionais; a rotação é usada quando uma chave esgota a cota diária. |
| `GROQ_RPM_BUDGET` | Teto local preventivo de chamadas ao Nó 3 por minuto (padrão: `10`). Ajuste para um valor igual ou inferior ao RPM da sua conta Groq. Não alterna chaves por chamada. |
| `NODE3_MOCK_MODE` | `false` por padrão; `true` gera dados de demonstração sem chamar a Groq. |
| `CHROMA_PERSIST_DIRECTORY` | Diretório do banco vetorial; padrão `./data/vector_db`. |
| `LICIT_CHECKPOINT_DIR` | Diretório de checkpoints locais; padrão `./data/checkpoints`. |

Copie `.env.example` para `.env`; o arquivo `.env` é ignorado pelo Git. O modelo da Groq e o modelo de embeddings são definidos no código dos respectivos nós. As variáveis antigas `DEFAULT_MODEL`, `CHROMA_COLLECTION_NAME` e `API_PORT` não controlam este fluxo.

Para conferir o modo ativo, abra `http://127.0.0.1:8002/status`. O modo de demonstração é sinalizado na interface e não deve ser usado como análise de um edital real.

## Como a análise funciona

1. **Nó 1 — PDF:** extrai texto com `pdfplumber` e divide o conteúdo em trechos, preservando o número físico da página. PDFs sem texto extraível podem exigir OCR antes do uso.
2. **Nó 2 — embeddings:** codifica os trechos com `sentence-transformers` e cria uma coleção ChromaDB separada para cada execução.
3. **Nó 3 — extração e explicação:** envia os trechos em lotes à Groq. Cada regra de participação ou seleção é extraída como um registro com título, trecho literal, página e eventual condição. O sistema confere que a citação existe no chunk indicado antes de salvá-la no checkpoint; se a resposta inventar uma fonte ou omitir o esquema, tenta o lote até três vezes e então falha sem publicá-lo. O contexto dos trechos anteriores da mesma página permite reconhecer condições que cruzem a fronteira entre lotes. A explicação usa a cláusula associada e o trecho anterior da mesma página. O resumo ainda usa busca temática no ChromaDB. As explicações percorrem todos os itens extraídos em lotes de cinco, com até duas novas tentativas quando faltar algum ID.
4. **Nó 4 — checklist:** organiza os candidatos a documentos para proposta ou habilitação. Documentos da execução, anexos fornecidos pelo órgão e itens de etapa incerta ficam separados. Se o nó 3 falhar, o nó 4 é ignorado.

O resumo usa uma amostra dos itens extraídos e informa sua cobertura. As explicações item a item percorrem todos os itens **identificados pela extração**. Isso não garante que todas as cláusulas do PDF tenham sido encontradas. Trechos recuperados por similaridade são pistas para verificação; não provam automaticamente cada conclusão.

As indicações **aplicável**, **condicional** e **regra alternativa** são classificações automáticas, não validação jurídica. Para regras novas, a página e a citação vêm da extração estruturada. Uma citação literal confirma que o trecho aparece no PDF, mas ainda não prova que o título ou a interpretação estejam corretos. A comparação entre uma modalidade condicional e o modo escolhido no cabeçalho é feita com os textos do próprio documento; uma referência a subitem anterior sem a condição preservada fica pendente. Números novos que não constam no título, na citação ou na condição impedem a publicação da interpretação como confirmada. Uma leitura preliminar não confirma as condições ou a aplicação jurídica da regra. Não há exceções programadas para um edital específico.

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

A Groq pode responder `429 Too Many Requests` ou falhar temporariamente com `503 Service Unavailable`. O nó 3 lê dos cabeçalhos de cada resposta as cotas restantes de tokens por minuto e requisições por dia, e aguarda a renovação quando o próximo pedido pode superar o saldo. Também limita localmente chamadas por minuto com `GROQ_RPM_BUDGET`. A estimativa prévia de tokens é conservadora, mas não equivale ao tokenizador da Groq; chamadas de outros processos e limites separados de entrada/saída ainda podem gerar `429`. Em falhas temporárias de conexão ou HTTP 500/502/503/504, o nó faz até três tentativas adicionais com esperas de 5, 15 e 30 segundos; se persistir, o checkpoint permite retomar sem repetir lotes concluídos.

Se ocorrer `429`, o nó 3 registra o tipo e o código do limite quando a Groq os fornecer, respeita `retry-after` quando disponível e faz até três novas tentativas na mesma chave. Depois do primeiro `429`, reduz o teto local para uma nova chamada por minuto enquanto esse job continuar. Quando a Groq informa explicitamente esgotamento da cota diária de tokens, tenta a próxima chave configurada; chaves da mesma organização não ampliam os limites globais. O job encerra com erro recuperável se a cota diária de requisições zerar, a espera necessária exceder cinco minutos ou o limite continuar após as tentativas. Um novo job do mesmo PDF retoma os dados salvos quando a cota voltar a estar disponível.

Após cada lote de extração e cada explicação válida, o nó 3 salva um checkpoint em `data/checkpoints/`. Se você iniciar **outra análise do mesmo PDF** com o mesmo modelo e a mesma versão do pipeline, ele reaproveita as chamadas concluídas e continua nos itens faltantes. A leitura do PDF e a indexação vetorial são executadas novamente. Se o conteúdo, o modelo ou a versão do checkpoint mudar, a extração começa de novo. Jobs da API existem apenas na memória do processo; o ID antigo deixa de funcionar após reiniciar a API, mas os checkpoints permanecem no disco.

**Atualização do esquema de fontes:** a versão 2 do checkpoint exige a origem de cada regra. O arquivo da versão anterior permanece no disco, mas não fornece essa informação; por isso a primeira análise após esta mudança faz novamente os lotes de extração, gerando novas chamadas à Groq. A nova versão salva cada lote e permite retomar uma execução interrompida. Não apague os checkpoints antigos para atualizar.

Quando apenas as instruções de redação mudam, a versão das explicações pode ser atualizada sem descartar os lotes de extração do mesmo esquema. A migração para o novo esquema de origem é diferente: os registros antigos não contêm cláusulas verificadas e exigem nova extração. O sistema ainda pode omitir cláusulas, extrair títulos errados ou interpretar mal uma condição mesmo com uma citação verdadeira. O relatório precisa ser confrontado com o PDF original.

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
