"""Regressões para o checklist do Nó 4."""

from app.rag.node_4_document_generator import Node4DocumentGenerator


def test_checklist_deduplicates_documents_without_assuming_obligation():
    generator = Node4DocumentGenerator(mock_mode=True)
    analysis = {
        "llm_analysis": {
            "documentos_exigidos": [
                "CNPJ", "Prova de inscrição no CNPJ", "Cronograma físico-financeiro",
                "Cronograma Físico Financeiro", "Balanço patrimonial",
                "Atestado de capacidade técnica", "Procuração",
            ],
            "prazos": [],
        },
        "structured_analysis": {},
    }

    result = generator.process_from_node3(analysis)
    assert result["resumo"]["total_documentos"] == 5
    assert result["resumo"]["obrigatorios"] == 0
    assert result["resumo"]["a_confirmar"] == 5
    assert result["resumo"]["opcionais"] == 0
    assert result["checklist"]["economica"]["itens"][0]["documento"] == "Balanço patrimonial"
    assert result["checklist"]["tecnica"]["itens"][0]["documento"] == "Atestado de capacidade técnica"
    assert result["checklist"]["proposta"]["itens"][0]["documento"] == "Cronograma físico-financeiro"
    assert all(
        item["obrigatorio"] is None
        for category in result["checklist"].values()
        for item in category["itens"]
    )


def test_juridical_category_does_not_match_ata_inside_other_words():
    generator = Node4DocumentGenerator(mock_mode=True)
    assert generator.categorize_document("Contrato da contratada") == "outros"
    assert generator.categorize_document("Contrato social") == "juridica"
