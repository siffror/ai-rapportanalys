# core/embedding_utils.py

from openai import OpenAI, OpenAIError
from functools import lru_cache
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
import streamlit as st
from typing import List

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

@retry(
    retry=retry_if_exception_type(OpenAIError),
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6)
)
@lru_cache(maxsize=512)
def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """
    Genererar en embedding-vektor för en given text via OpenAI:s API.

    Args:
        text (str): Texten som ska konverteras till embedding.
        model (str): Modellnamn för OpenAI embeddings (default: "text-embedding-3-small").

    Returns:
        List[float]: En lista som representerar embedding-vektorn.
    """
    if not text:
        raise ValueError("Text för embedding får inte vara tom.")
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding
