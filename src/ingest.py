import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from langchain_core.embeddings import Embeddings

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH", "document.pdf")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/rag")
PG_VECTOR_COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "pdf_documents")


def get_embeddings() -> Embeddings:
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    google_key = os.getenv("GOOGLE_API_KEY", "").strip()

    if provider == "google" or (not openai_key and google_key):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        model_name = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001").strip().strip("'\"")
        return GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=google_key)

    if provider == "openai" or openai_key:
        from langchain_openai import OpenAIEmbeddings

        model_name = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip().strip("'\"")
        return OpenAIEmbeddings(model=model_name, api_key=openai_key)

    raise ValueError(
        "Nenhuma chave de API configurada. Defina OPENAI_API_KEY ou GOOGLE_API_KEY no arquivo .env."
    )


def ingest_pdf(pdf_path: str = PDF_PATH) -> int:
    resolved_path = Path(pdf_path)
    if not resolved_path.exists():
        fallback_path = Path(__file__).resolve().parent.parent / pdf_path
        if fallback_path.exists():
            resolved_path = fallback_path
        else:
            raise FileNotFoundError(f"Arquivo PDF não encontrado: {pdf_path}")

    print(f"Carregando documento: {resolved_path}")
    loader = PyPDFLoader(str(resolved_path))
    documents = loader.load()
    print(f"Documento carregado. Total de páginas: {len(documents)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=False,
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        print("Nenhum chunk gerado para o documento.")
        return 0

    total_chunks = len(chunks)
    print(f"Documento dividido em {total_chunks} chunks (tamanho=1000, overlap=150).")

    embeddings = get_embeddings()
    print(f"Conectando ao banco PostgreSQL e collection '{PG_VECTOR_COLLECTION_NAME}'...")

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=PG_VECTOR_COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )

    ids = [f"doc-{i}" for i in range(total_chunks)]
    batch_size = int(os.getenv("INGEST_BATCH_SIZE", "10"))
    print(f"Inserindo {total_chunks} vetores no pgVector em lotes de {batch_size}...")

    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        current_batch = i // batch_size + 1
        total_batches = (total_chunks + batch_size - 1) // batch_size
        print(f"Inserindo lote {current_batch}/{total_batches} (chunks {i + 1} a {min(i + batch_size, total_chunks)})...")

        max_retries = 4
        for attempt in range(max_retries):
            try:
                vector_store.add_documents(documents=batch_chunks, ids=batch_ids)
                break
            except Exception as err:
                err_str = str(err)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait_time = (attempt + 1) * 4
                    print(f"Aviso de quota (429). Aguardando {wait_time}s antes de tentar novamente...")
                    time.sleep(wait_time)
                else:
                    raise err
        else:
            raise RuntimeError(f"Falha ao inserir lote {current_batch} após {max_retries} tentativas.")

        time.sleep(0.5)

    print("Ingestão finalizada com sucesso!")
    return total_chunks


if __name__ == "__main__":
    try:
        ingest_pdf()
    except Exception as err:
        print(f"Erro durante a ingestão: {err}", file=sys.stderr)
        sys.exit(1)