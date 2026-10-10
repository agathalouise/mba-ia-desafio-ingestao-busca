import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from search import PROMPT_TEMPLATE, QARunner


def test_pdf_chunking_spec():
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    pdf_path = Path(__file__).resolve().parent.parent / "document.pdf"
    assert pdf_path.exists(), "document.pdf deve existir na raiz do projeto"

    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    assert len(docs) > 0, "O documento PDF deve conter páginas"

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    assert len(chunks) > 0, "O splitter deve gerar chunks"

    for chunk in chunks:
        assert len(chunk.page_content) <= 1200


def test_prompt_template_structure():
    assert "CONTEXTO:" in PROMPT_TEMPLATE
    assert "{contexto}" in PROMPT_TEMPLATE
    assert "REGRAS:" in PROMPT_TEMPLATE
    assert "PERGUNTA DO USUÁRIO:" in PROMPT_TEMPLATE
    assert "{pergunta}" in PROMPT_TEMPLATE
    assert "Não tenho informações necessárias para responder sua pergunta." in PROMPT_TEMPLATE


def test_qa_runner_execution_and_invocation():
    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search_with_score.return_value = [
        (Document(page_content="SuperTechIABrazil faturamento: 10 milhões de reais em 2023."), 0.15),
        (Document(page_content="Fundada em 2020 no setor de inteligência artificial."), 0.25),
    ]

    fake_chat = FakeListChatModel(
        responses=[
            "O faturamento foi de 10 milhões de reais.",
            "O faturamento foi de 10 milhões de reais.",
            "O faturamento foi de 10 milhões de reais.",
        ]
    )

    runner = QARunner(mock_vector_store, fake_chat)

    # Teste com chamada direta .run()
    res1 = runner.run("Qual o faturamento da Empresa SuperTechIABrazil?")
    assert "10 milhões" in res1
    mock_vector_store.similarity_search_with_score.assert_called_with(
        "Qual o faturamento da Empresa SuperTechIABrazil?", k=10
    )

    # Teste com .invoke(str)
    res2 = runner.invoke("Qual o faturamento?")
    assert "10 milhões" in res2

    # Teste com .invoke(dict)
    res3 = runner.invoke({"pergunta": "Qual o faturamento?"})
    assert "10 milhões" in res3


def test_qa_runner_outside_context():
    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search_with_score.return_value = [
        (Document(page_content="SuperTechIABrazil faturamento: 10 milhões de reais."), 0.90)
    ]

    expected_out_of_context = "Não tenho informações necessárias para responder sua pergunta."
    fake_chat = FakeListChatModel(responses=[expected_out_of_context])

    runner = QARunner(mock_vector_store, fake_chat)
    res = runner.run("Quantos clientes temos em 2024?")
    assert res == expected_out_of_context
