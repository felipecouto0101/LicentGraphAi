# Guardrails de documentos

O cliente Gemini compartilhado pelo mapa e pela análise completa trata PDFs,
trechos recuperados e respostas anteriores como dados não confiáveis. As regras
da aplicação ficam em uma mensagem de sistema. As requisições com fontes ficam
em mensagens de usuário codificadas em JSON; texto do PDF não cria mensagens de
sistema ou de ferramenta, mesmo que contenha delimitadores semelhantes.

As chamadas não oferecem ferramentas, execução de código, pesquisa ou funções.
O SDK recebe `tools=[]`, modo de funções `NONE` e chamada automática de funções
desativada. Respostas com chamadas de ferramentas ou blocos de execução são
rejeitadas sem publicar argumentos e sem repetir a mesma ação. A chave da API
é usada pelo transporte; não é colocada nas mensagens para o modelo.

## Validação da saída

Contratos Pydantic locais rejeitam campos desconhecidos, tipos incorretos e
listas/textos acima dos limites no mapa e nas respostas estruturadas da análise
completa. As verificações de IDs, linhas, cobertura, páginas e citações continuam
sendo feitas no código de domínio. Regras sem fonte literal verificável mantêm
o tratamento de revisão existente; um contrato correto não comprova veracidade.
O checklist é montado localmente a partir dos resultados validados.

## Aviso de revisão

Uma inspeção local identifica alguns padrões de alteração de instruções,
simulação de mensagens de sistema e solicitação de credenciais. Detecta também
quebras de linha e alguns caracteres invisíveis. O resultado da API inclui
`security_warnings`, com páginas físicas e códigos de sinais. A interface do mapa
mostra as páginas para conferência, inclusive quando a organização é interrompida.
O texto original permanece intacto; o aviso não comprova um ataque nem descarta
automaticamente a cláusula. Exigências como “o candidato deverá apresentar
identidade” continuam sendo analisadas.

Eventos `guardrail.document_flagged` e `guardrail.response_rejected` registram
somente contagens, operação e motivo, sem copiar texto suspeito, argumentos ou
credenciais para a telemetria. As validações não fazem chamadas adicionais à IA.
As instruções de proteção acrescentam um pequeno contexto às chamadas existentes;
a estimativa local de tokens inclui esse contexto.

## Testes e limites

O CI verifica separação de mensagens, configuração do SDK real com transporte
simulado, bloqueio de ações, contratos, leitura de um PDF sintético, preservação
de fontes e aviso na API/interface. Esses testes não medem a resistência real
do modelo. Para uma avaliação opcional com o Gemini configurado no `.env`:

```bash
python -m scripts.evaluate_guardrails
```

O comando envia quatro casos sintéticos, consome cota e pode gerar custo. Imprime
somente PASS/FAIL e contagens; uma falha pode indicar indisponibilidade ou saída
diferente do resultado esperado. Não faz parte do CI e não é executado
automaticamente. Passar nesses casos não garante resistência a outras variações.

Regex, prompts e contratos reduzem riscos, mas não detectam todas as instruções
maliciosas nem garantem a correção semântica das respostas. Checkpoints existentes
permanecem disponíveis; as novas regras não comprovam retroativamente a segurança
das respostas já salvas. O modelo continua sem capacidades para agir sobre o
computador ou serviços externos durante a análise documental.
