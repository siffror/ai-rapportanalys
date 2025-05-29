
import logging
from typing import List, Tuple, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

from core.embedding_utils import get_embedding
from core.chunking import chunk_text

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- System prompts ---
ADVANCED_ANALYSIS_SYSTEM_PROMPT_SV = (
    "Du är en avancerad AI-baserad finansanalytiker och fondförvaltare. "
    "När du får en årsrapport eller ekonomisk text ska du göra en noggrann analys baserat endast på den kontext du får. "
    "Följ dessa steg:\n"
    "1. **Lönsamhetsanalys:** Analysera företagets lönsamhet, nyckeltal (t.ex. vinstmarginal, avkastning på eget kapital, rörelseresultat, kassaflöde, utdelning etc). Motivera tydligt med siffror och citat från kontexten.\n"
    "2. **Riskbedömning:** Identifiera risker och osäkerhetsfaktorer som nämns i texten. Tydliggör eventuella varningar.\n"
    "3. **Aktie- eller fondrekommendation:** Ge en motiverad rekommendation: Köp, Behåll eller Sälj (eller rekommendation gällande fonder om relevant). Motivera utifrån analysen – ange alltid tydligt vad rekommendationen baseras på, och säg tydligt om underlaget är svagt!\n"
    "4. **Struktur:** Dela upp svaret i rubriker: Sammanfattning, Lönsamhet, Risker, Rekommendation.\n"
    "5. **Språk:** Svara alltid på det språk som används i frågan eller i kontexten (svenska, engelska, etc).\n"
    "6. **Källhänvisning:** Om möjligt, citera direkt ur kontexten (med citationstecken) så att användaren kan följa ditt resonemang.\n"
    "Du får aldrig gissa eller lägga till information som inte uttryckligen står i den tillhandahållna kontexten. Om kontexten är för svag, påpeka det tydligt."
)

ADVANCED_ANALYSIS_SYSTEM_PROMPT_EN = (
    "You are an advanced AI-based financial analyst and fund manager. "
    "When you receive an annual report or financial text, perform a thorough analysis based solely on the provided context. "
    "Follow these steps:\n"
    "1. **Profitability Analysis:** Analyze the company's profitability, key metrics (e.g., profit margin, return on equity, operating income, cash flow, dividend, etc.). Justify your conclusions clearly with numbers and quotes from the context.\n"
    "2. **Risk Assessment:** Identify any risks and uncertainties mentioned in the text. Clearly highlight any warnings.\n"
    "3. **Stock or Fund Recommendation:** Give a justified recommendation: Buy, Hold, or Sell (or recommendation regarding funds if relevant). Clearly explain what your recommendation is based on, and indicate if the available information is weak.\n"
    "4. **Structure:** Divide the response into sections: Summary, Profitability, Risks, Recommendation.\n"
    "5. **Language:** Always answer in the same language as the user's question or the context (English, Swedish, etc).\n"
    "6. **Source citation:** If possible, quote directly from the context (using quotation marks) so that the user can follow your reasoning.\n"
    "Never guess or add information not explicitly found in the provided context. If the context is too weak, state this clearly."
)

def detect_language(text: str) -> str:
    """
    Enkel språkdetection – utöka gärna med t.ex. langdetect om du vill ha säkrare resultat!
    """
    # Väldigt enkel logik: Kolla om många svenska ord finns, annars default engelska
    swedish_keywords = ['och', 'eller', 'företag', 'utdelning', 'omsättning', 'resultat', 'kassaflöde', 'vinst', 'risker', 'rekommendation', 'aktie', 'köp', 'sälj']
    text_lower = text.lower()
    hits = sum(word in text_lower for word in swedish_keywords)
    return "sv" if hits > 1 else "en"

def get_system_prompt(language: str) -> str:
    if language == "sv":
        return ADVANCED_ANALYSIS_SYSTEM_PROMPT_SV
    else:
        return ADVANCED_ANALYSIS_SYSTEM_PROMPT_EN

def search_relevant_chunks(
    question: str,
    embedded_chunks: List[Dict[str, Any]],
    top_k: int = 7
) -> Tuple[str, List[Tuple[float, str]]]:
    query_embed = get_embedding(question)
    question_words = set(question.lower().split())
    similarities = []
    for item in embedded_chunks:
        text = item.get("text", "")
        text_lower = text.lower()
        score = cosine_similarity([query_embed], [item["embedding"]])[0][0]
        fuzzy_bonus = sum(1 for word in question_words if word in text_lower) * 0.005
        score += fuzzy_bonus
        similarities.append((score, text))
    top_chunks = sorted(similarities, key=lambda x: x[0], reverse=True)[:top_k]
    context = "\n---\n".join([chunk for _, chunk in top_chunks])
    logger.info(f"Valde top {top_k} chunks för frågan.")
    return context, top_chunks

def generate_gpt_answer(
    question: str,
    context: str,
    model: str = "gpt-4o",
    temperature: float = 0.3,
    max_tokens: int = 1000,
    language: str = None
) -> str:
    """
    Skapar ett GPT-svar på rätt språk och med avancerad prompt.
    Om 'language' är None, autodetekteras språk baserat på fråga + kontext.
    """
    from openai import OpenAI, OpenAIError
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    if not context.strip():
        raise ValueError("Kontext får inte vara tom vid generering.")
    if language is None:
        language = detect_language(question + " " + context)
    system_prompt = get_system_prompt(language)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Kontext:\n{context}\n\nFråga: {question}"}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except OpenAIError as e:
        logger.error(f"OpenAI API-fel: {e}")
        raise RuntimeError(f"❌ Fel vid generering av svar: {e}")

def full_rapportanalys(
    text: str,
    model: str = "gpt-4o",
    temperature: float = 0.3,
    max_tokens: int = 1500,
    language: str = None
) -> str:
    """
    Gör en fullständig rapportanalys med avancerad prompt.
    """
    from openai import OpenAI
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    if language is None:
        language = detect_language(text)
    system_prompt = get_system_prompt(language)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI API-fel vid analys: {e}")
        return f"❌ Fel vid analys: {e}"
