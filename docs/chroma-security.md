# Segurança do ChromaDB local

O projeto mantém **ChromaDB 1.5.9**, com persistência e índice HNSW existentes.
Não há versão estável corrigida informada para os quatro avisos abaixo na revisão
de 01/10/2026. A biblioteca não é declarada corrigida nem livre de vulnerabilidades.

## Controles da aplicação

`app/rag/chroma_security.py` centraliza a construção dos clientes. O backend é
explicitamente `chromadb.api.rust.RustBindingsAPI`, inclusive quando variáveis de
ambiente tentam selecionar HTTP, backend Python ou provedores de autorização.
Um backend diferente ou versão não revisada causa erro. Telemetria é desativada.

O Nó 2 calcula os embeddings com Sentence Transformers e entrega vetores ao
Chroma. Coleções são criadas e abertas com `embedding_function=None`. Uma fachada
permite somente adicionar/atualizar com embeddings explícitos e consultar por
vetores; consultas por texto, URI e funções persistidas personalizadas são
rejeitadas. Isso também bloqueia o fallback de embedding por schema da versão
1.5.9, que pode existir mesmo com `embedding_function=None`.

Coleções antigas com configuração padrão continuam legíveis sem executar essa
função. Não é preciso reconstruir o índice. PDFs, índices e checkpoints permanecem
no formato atual. Diretórios de banco devem ser controlados pela aplicação;
carregar bancos fornecidos por terceiros está fora desse escopo de segurança.

## Aplicabilidade revisada

| Aviso | Caminho descrito | Tratamento neste projeto |
| --- | --- | --- |
| PYSEC-2026-311 / CVE-2026-45829 | Criação de coleção no servidor Python com modelo remoto | Servidor ausente; backend local e operações somente com vetores. |
| PYSEC-2026-3814 / CVE-2026-45833 | Alteração de coleção no servidor Python com modelo remoto | Servidor ausente; funções personalizadas e embeddings implícitos bloqueados. |
| PYSEC-2026-3813 / CVE-2026-45830 | Acesso autenticado a coleções de outro tenant no servidor | Não existe API Chroma exposta a usuários/tenants. |
| PYSEC-2026-3815 / CVE-2026-45831 | SimpleRBAC sem validação do recurso | Provedor ausente e desativado na configuração do cliente. |

Essas conclusões valem para a aplicação local, com banco confiável e sem servidor
Chroma separado. Não se aplicam a `chroma run`, servidores HTTP, Chroma Cloud ou
serviço compartilhado entre usuários. A API da aplicação não oferece endpoints
para configurar funções de embedding ou modificar configurações do Chroma.

## CI e revisão

O job exporta todos os pacotes instalados, inclusive transitivos e ferramentas,
para um manifesto fixado. O build oficial PyTorch `+cpu` é consultado pela versão
pública correspondente; o inventário preserva a versão instalada e esse mapeamento.
Outros builds locais são recusados. Não há exclusão de pacotes nem nova resolução
de dependências durante a consulta.

O pip-audit gera o **relatório bruto completo**, incluindo os quatro alertas.
Não é usado `--ignore-vuln`. O job executa testes do Chroma real antes da auditoria.
`security/dependency-reviews.json` mantém a análise dos quatro avisos como
informação de contexto; não controla a aprovação nem expira o pipeline.

A política vale para todos os pacotes: avisos sem versão corrigida informada pela
base consultada são publicados como warnings e permitem aprovação. Avisos com
correção disponível bloqueiam o check. Falhas da auditoria, pacotes não auditados,
conflitos de dependências e falhas dos testes continuam bloqueando. A política
não tem prazo de expiração; a análise é executada novamente a cada run.

A aprovação não significa que a biblioteca foi corrigida ou que os riscos foram
eliminados. O relatório bruto e os avisos continuam disponíveis nos artefatos e
no resumo do job, incluindo novos avisos sem correção.

Os testes usam o Chroma real com vetores determinísticos; não baixam modelos nem
consomem APIs de IA. Os testes que carregam Sentence Transformers real permanecem
separados como integração.

Referências: [RCE na criação](https://github.com/advisories/GHSA-f4j7-r4q5-qw2c),
[RCE na alteração](https://github.com/advisories/GHSA-36p7-vc44-83pf),
[acesso entre tenants](https://osv.dev/vulnerability/PYSEC-2026-3813),
[SimpleRBAC](https://github.com/advisories/GHSA-xph7-9rjv-w5fr),
[análise dos caminhos Python/Rust](https://github.com/chroma-core/chroma/issues/7588).
