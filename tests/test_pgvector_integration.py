import os
from langchain_core.documents import Document
from langchain_core.embeddings.fake import FakeEmbeddings
from langchain_postgres import PGVector


def test_pgvector_connection_and_search():
    database_url = os.getenv("DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/rag")
    collection_name = "test_collection_ci"

    embeddings = FakeEmbeddings(size=1536)

    store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=database_url,
        use_jsonb=True,
        pre_delete_collection=True,
    )

    docs = [
        Document(page_content=f"Empresa Teste {i} faturamento de {i * 10} milhões de reais.", metadata={"id": i})
        for i in range(15)
    ]
    ids = [f"test-doc-{i}" for i in range(15)]

    store.add_documents(documents=docs, ids=ids)

    # Verifica busca com k=10
    results = store.similarity_search_with_score("faturamento", k=10)
    assert len(results) == 10

    for doc, score in results:
        assert isinstance(doc, Document)
        assert doc.page_content.startswith("Empresa Teste")
        assert isinstance(score, float)
