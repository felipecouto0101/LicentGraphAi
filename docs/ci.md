# Integração contínua

O workflow `CI` executa em pull requests para `main`, pushes em `main`
e por acionamento manual. Os jobs independentes rodam em
paralelo. Execuções anteriores da mesma referência são canceladas quando uma nova
execução começa.

## Verificações

| Check | Escopo | Condição de falha |
| --- | --- | --- |
| Ruff | `app/`, `tests/` e `start_app.py` | Diagnósticos de sintaxe e Pyflakes (`E9`, `F`), incluindo nomes indefinidos e imports não utilizados. |
| Tests (Python 3.12) | Testes sem a marca `integration`, incluindo API, fontes, clientes simulados e interface | Qualquer teste selecionado falha ou conflito nas dependências de teste. |
| Bandit | Código da aplicação e inicializador | Achados de severidade média/alta e confiança média/alta. |
| Semgrep | Regras Community Edition `p/python` e `p/security-audit` | Achados classificados como `ERROR` ou erros de análise no modo estrito. |
| Dependency audit | Ambiente instalado a partir de `requirements.txt`, incluindo dependências transitivas | Vulnerabilidades conhecidas encontradas pelo pip-audit, falha na consulta ou conflito de dependências. |
| Gitleaks | Histórico Git completo disponível no checkout | Segredos identificados ou erro na execução do scanner. |
| CI gate | Resultado de todos os jobs | Qualquer check falha, é cancelado ou fica incompleto. |

Os relatórios JSON, SARIF e JUnit ficam nos **Artifacts** da execução por sete
dias, inclusive quando a análise falha. O Gitleaks oculta os valores dos segredos
no relatório. O Semgrep publica SARIF como arquivo; a configuração não depende do
recurso pago de code scanning para repositórios privados.

O lint verifica erros de código; a formatação não é uma condição de aprovação.
Achados de menor severidade permanecem nos relatórios para revisão. Exceções
pontuais devem identificar a regra e explicar por que o alerta não representa
uma vulnerabilidade; não há uma baseline global que oculte problemas existentes.

## Ambiente

Os testes usam `requirements-ci.txt`, com versões fixadas para Python 3.12. Não
recebem chaves Gemini/Groq nem baixam modelos de embeddings. O teste de startup
bloqueia imports de dependências pesadas e verifica os endpoints reais da API.

`tests/test_node_2.py` contém testes de integração que carregam modelos reais do
Hugging Face e SQLite. Eles estão marcados como `integration` e não fazem parte
do job offline. Para executá-los em um ambiente com as dependências completas e
acesso aos modelos, use `python -m pytest tests -m integration`.

A auditoria de dependências usa o manifesto completo `requirements.txt`, em um
job separado. PyTorch é instalado pela distribuição CPU para evitar pacotes CUDA.
O relatório corresponde às versões resolvidas nessa execução; ele não representa
automaticamente o ambiente instalado na máquina de cada desenvolvedor.

As actions são fixadas por SHA, o token tem apenas leitura do repositório e as
credenciais do checkout não são persistidas. Os scanners não fazem chamadas aos
provedores de IA. Semgrep baixa regras do registro e pip-audit consulta bases de
vulnerabilidades. Os checks possuem limites de duração e cache de pacotes.

## Uso local

```bash
python -m pip install -r requirements-ci.txt
python -m pip install ruff==0.16.9 bandit==1.9.4
ruff check app tests start_app.py
python -m pytest tests -m "not integration"
bandit -r app start_app.py --severity-level medium --confidence-level medium
```

Instale Semgrep em um ambiente separado das dependências da aplicação:

```bash
semgrep scan --config p/python --config p/security-audit --metrics off app start_app.py
```

Para auditar o ambiente completo da aplicação, instale `pip-audit==2.10.1` nesse
ambiente e execute `python -m pip_audit --local`.

## Proteção da branch

O workflow publica o check `CI gate`. A exigência desse check antes do merge é uma
configuração do repositório em **Settings → Rules → Rulesets** ou **Branches**,
conforme o plano do GitHub. O arquivo de workflow, sozinho, não impede o merge.

Os checks analisam código e dependências; não comprovam a ausência de todas as
vulnerabilidades nem avaliam a qualidade semântica das respostas da IA.

## Correção da dependência vetorial

A auditoria identificou quatro avisos no ChromaDB sem versões corrigidas
informadas. A aplicação removeu essa dependência: vetores, documentos e metadados
são persistidos em SQLite, com busca exata por cosseno usando NumPy. Não foram
adicionadas exceções ao pip-audit; a auditoria continua cobrindo o ambiente completo.

Os testes offline exercitam o armazenamento real, persistência, isolamento entre
coleções, ranking, metadados, validação e escrita atômica. Os testes de integração
separados ainda verificam o modelo de embeddings real. Para atualizar ambientes
locais, use um ambiente virtual novo: instalar o novo manifesto em um ambiente
antigo não desinstala automaticamente o ChromaDB.

O PyPDF2 não utilizado foi removido. O job atualiza pip e setuptools antes da
resolução para evitar versões antigas das ferramentas pré-instaladas no runner.
