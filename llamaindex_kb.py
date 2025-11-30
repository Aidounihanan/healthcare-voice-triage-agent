# llamaindex_kb.py
import os
from pathlib import Path
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# Charger les variables d'environnement
load_dotenv()

_triage_index = None


def _build_index() -> VectorStoreIndex:
    """
    Build a VectorStoreIndex from local markdown guidelines.
    Uses OpenAI for embeddings and LLM.
    """
    # Configuration globale pour utiliser OpenAI
    Settings.llm = OpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    data_dir = Path(__file__).parent / "data"
    
    # Vérifier que le dossier existe
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    documents = SimpleDirectoryReader(str(data_dir)).load_data()
    
    if not documents:
        raise ValueError(f"No documents found in {data_dir}")

    index = VectorStoreIndex.from_documents(documents)
    return index


def get_triage_answer(query: str) -> str:
    """
    Query the triage guidelines and return a plain answer.
    """
    global _triage_index
    if _triage_index is None:
        _triage_index = _build_index()
    query_engine = _triage_index.as_query_engine()
    response = query_engine.query(query)
    return str(response)