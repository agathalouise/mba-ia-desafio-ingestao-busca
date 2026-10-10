import os
from typing import Any
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_postgres import PGVector

try:
    from ingest import get_embeddings, DATABASE_URL, PG_VECTOR_COLLECTION_NAME
except ModuleNotFoundError:
    from src.ingest import get_embeddings, DATABASE_URL, PG_VECTOR_COLLECTION_NAME

load_dotenv()

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def get_chat_model() -> BaseChatModel:
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    google_key = os.getenv("GOOGLE_API_KEY", "").strip()

    if provider == "google" or (not openai_key and google_key):
        from langchain_google_genai import ChatGoogleGenerativeAI

        model_name = os.getenv("GOOGLE_CHAT_MODEL", "gemini-3.8-flash").strip().strip("'\"")
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=google_key,
            temperature=0,
        )

    if provider == "openai" or openai_key:
        from langchain_openai import ChatOpenAI

        model_name = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        return ChatOpenAI(
            model=model_name,
            api_key=openai_key,
            temperature=0,
        )

    raise ValueError(
        "Nenhuma chave de API configurada. Defina OPENAI_API_KEY ou GOOGLE_API_KEY no arquivo .env."
    )


def get_vector_store() -> PGVector:
    embeddings = get_embeddings()
    return PGVector(
        embeddings=embeddings,
        collection_name=PG_VECTOR_COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )


class QARunner:
    def __init__(self, vector_store: PGVector, chat_model: BaseChatModel):
        self.vector_store = vector_store
        self.chat_model = chat_model
        self.prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
        self.generation_chain = self.prompt | self.chat_model | StrOutputParser()

    def run(self, question: str) -> str:
        results = self.vector_store.similarity_search_with_score(question, k=10)
        context_parts = [doc.page_content.strip() for doc, _score in results if doc.page_content]
        contexto = "\n\n---\n\n".join(context_parts)

        return self.generation_chain.invoke({
            "contexto": contexto,
            "pergunta": question,
        })

    def invoke(self, input_val: Any) -> str:
        if isinstance(input_val, dict):
            question = str(input_val.get("pergunta") or input_val.get("question") or "")
        else:
            question = str(input_val)
        return self.run(question)

    def __call__(self, question: str) -> str:
        return self.run(question)


def search_prompt(question: str | None = None) -> Any:
    try:
        vector_store = get_vector_store()
        chat_model = get_chat_model()
        runner = QARunner(vector_store, chat_model)
    except Exception as err:
        print(f"Erro ao inicializar search_prompt: {err}")
        return None

    if question is not None:
        return runner.run(question)

    return runner