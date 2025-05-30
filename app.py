import streamlit as st
from dotenv import load_dotenv

from core.gpt_logic import (
    search_relevant_chunks, generate_gpt_answer, get_embedding,
    chunk_text, full_rapportanalys
)
from core.file_processing import extract_text_from_file
from utils.cache_utils import get_embedding_cache_name, save_embeddings, load_embeddings_if_exists
from utils.general import is_key_figure
from utils.ocr_utils import extract_text_from_image_or_pdf
from utils.pdf_utils import answer_to_pdf
from utils.file_utils import save_output_file, save_uploaded_file
from services.html_downloader import fetch_html_text

import os
import requests
from streamlit_lottie import st_lottie  # <-- Glöm inte denna!

# --- Skapa nödvändiga datamappar om de inte finns ---
for d in ["data/embeddings", "data/outputs", "data/uploads"]:
    os.makedirs(d, exist_ok=True)

load_dotenv()
st.set_page_config(page_title="📊 AI Rapportanalys", layout="wide")

# --- Lottie-animation (från din nya kod) ---
col1, col2, col3 = st.columns([3, 4, 3])
with col2:
    lottie_url = "https://raw.githubusercontent.com/siffror/ai-rapportanalys/main/1GRWuk0lXN.json"
    r = requests.get(lottie_url)
    if r.status_code == 200:
        lottie_json = r.json()
        st_lottie(
            lottie_json,
            speed=1,
            width=240,
            height=240,
            loop=True,
            quality="high",
            key="ai_logo"
        )
    else:
        st.warning("Kunde inte ladda AI-animationen.")

st.markdown("<h1 style='color:#3EA6FF;'>📊 AI-baserad Rapportanalys</h1>", unsafe_allow_html=True)
st.image("https://www.appypie.com/dharam_design/wp-content/uploads/2025/05/headd.svg", width=120)

# --- Input: HTML-länk, uppladdning, eller manuell text ---
html_link = st.text_input("🌐 Rapport-länk (HTML)")
uploaded_file = st.file_uploader("📎 Ladda upp HTML, PDF, bild eller text", 
    type=["html", "pdf", "txt", "xlsx", "xls", "png", "jpg", "jpeg"])

preview, ocr_text = "", ""

# --- Hantera filuppladdning och ev. spara till server ---
if uploaded_file:
    save_path = save_uploaded_file(uploaded_file)
    st.info(f"Uppladdad fil har sparats som: {save_path}")

    # Extrahera text beroende på filtyp
    if uploaded_file.name.endswith((".png", ".jpg", ".jpeg")):
        ocr_text, _ = extract_text_from_image_or_pdf(uploaded_file)
        st.text_area("📄 OCR-utläst text:", ocr_text[:2000], height=200)
    else:
        preview = extract_text_from_file(uploaded_file)
elif html_link:
    preview = fetch_html_text(html_link)
else:
    preview = st.text_area("✏️ Klistra in text manuellt här:", "", height=200)

# --- Kombinera extraherad text ---
text_to_analyze = preview or ocr_text

if preview:
    st.text_area("📄 Förhandsvisning:", preview[:5000], height=200)
else:
    st.warning("❌ Ingen text att analysera än.")

# --- Fullständig rapportanalys ---
if st.button("🔍 Fullständig rapportanalys"):
    if text_to_analyze and len(text_to_analyze.strip()) > 20:
        with st.spinner("📊 GPT analyserar hela rapporten..."):
            st.markdown("### 🧾 Fullständig AI-analys:")
            ai_report = full_rapportanalys(text_to_analyze)
            st.markdown(ai_report)

            # --- Spara analys som PDF på servern vid knapptryck ---
            if st.button("📄 Spara AI-analys som PDF på servern"):
                pdf_bytes = answer_to_pdf(ai_report)
                output_path = save_output_file("ai_full_analys.pdf", pdf_bytes)
                st.success(f"PDF-filen har sparats till servern: {output_path}")
    else:
        st.error("Ingen text tillgänglig för analys.")

# --- Frågebaserad GPT-analys ---
if "user_question" not in st.session_state:
    st.session_state.user_question = "Vilken utdelning per aktie föreslås?"
st.text_input("Fråga:", key="user_question")

if text_to_analyze and len(text_to_analyze.strip()) > 20:
    if st.button("🔍 Analysera med GPT"):
        with st.spinner("🤖 GPT analyserar..."):
            source_id = (html_link or uploaded_file.name if uploaded_file else text_to_analyze[:50]) + "-v2"
            cache_file = get_embedding_cache_name(source_id)
            embedded_chunks = load_embeddings_if_exists(cache_file)

            if not embedded_chunks:
                chunks = chunk_text(text_to_analyze)
                embedded_chunks = []
                for i, chunk in enumerate(chunks, 1):
                    st.write(f"🔹 Chunk {i} – {len(chunk)} tecken")
                    try:
                        embedding = get_embedding(chunk)
                        embedded_chunks.append({"text": chunk, "embedding": embedding})
                    except Exception as e:
                        st.error(f"❌ Fel vid embedding av chunk {i}: {e}")
                        st.stop()
                save_embeddings(cache_file, embedded_chunks)

            context, top_chunks = search_relevant_chunks(
                st.session_state.user_question, embedded_chunks)
            st.code(context[:1000], language="text")
            answer = generate_gpt_answer(st.session_state.user_question, context)
            st.success("✅ Svar klart!")
            st.markdown(f"### 🤖 GPT-4o svar:\n{answer}")

            key_figures = [row for row in answer.split("\n") if is_key_figure(row)]
            if key_figures:
                st.markdown("### 📊 Möjliga nyckeltal i svaret:")
                for row in key_figures:
                    st.markdown(f"- {row}")

            # --- Download/export (lägg till key på båda) ---
            st.download_button(
                "💾 Ladda ner svar (.txt)", 
                answer, 
                file_name="gpt_svar.txt", 
                key="dl_gpt_txt_main"
            )
            st.download_button(
                "📄 Ladda ner svar (.pdf)", 
                answer_to_pdf(answer), 
                file_name="gpt_svar.pdf", 
                key="dl_gpt_pdf_main"
            )

            # --- Spara GPT-svar som PDF på servern ---
            if st.button("📄 Spara GPT-svar som PDF på servern"):
                pdf_bytes = answer_to_pdf(answer)
                output_path = save_output_file("gpt_svar.pdf", pdf_bytes)
                st.success(f"PDF-filen har sparats till servern: {output_path}")
else:
    st.info("📝 Ange text, länk eller ladda upp en fil eller bild för att börja.")
