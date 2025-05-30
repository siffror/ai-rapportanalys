import streamlit as st
from dotenv import load_dotenv
import os
import requests # För Lottie

# RAGAS och LLM funktioner
from utils.evaluation_utils import ragas_evaluate
from core.gpt_logic import (
    search_relevant_chunks,
    generate_gpt_answer,
    chunk_text,
    full_rapportanalys
)
from core.embedding_utils import get_embedding

# Fil- och datahantering
from core.file_processing import extract_text_from_file
from utils.cache_utils import get_embedding_cache_name, save_embeddings, load_embeddings_if_exists
from utils.ocr_utils import extract_text_from_image_or_pdf
from utils.pdf_utils import answer_to_pdf
from utils.file_utils import save_output_file, save_uploaded_file
from services.html_downloader import fetch_html_text

# För Lottie-animation
from streamlit_lottie import st_lottie
import traceback # För detaljerad felutskrift

# --- Skapa nödvändiga datamappar ---
for d in ["data/embeddings", "data/outputs", "data/uploads"]:
    os.makedirs(d, exist_ok=True)

load_dotenv()
st.set_page_config(page_title="🤖 AI Rapportanalys", layout="wide")

# --- UI Start ---
col1_lottie, col2_lottie, col3_lottie = st.columns([3, 4, 3])
with col2_lottie:
    lottie_url = "https://raw.githubusercontent.com/siffror/ai-rapportanalys/main/1GRWuk0lXN.json"
    try:
        r = requests.get(lottie_url, timeout=10)
        r.raise_for_status()
        lottie_json = r.json()
        st_lottie(lottie_json, speed=1, width=240, height=240, loop=True, quality="high", key="ai_logo")
    except requests.exceptions.RequestException as e_req:
        st.warning(f"Kunde inte ladda AI-animationen (nätverksfel): {e_req}")
    except requests.exceptions.JSONDecodeError as e_json_lottie:
         st.warning(f"Kunde inte tolka AI-animationen (JSON-fel): {e_json_lottie}")

st.markdown("<h1 style='color:#3EA6FF;'>🤖 AI-baserad Rapportanalys</h1>", unsafe_allow_html=True)

# --- Sidomenyn är borttagen ---

# --- Huvudinnehåll: Input för rapportanalys ---
st.header("📄 Mata in rapporttext")
input_col1, input_col2 = st.columns(2)
with input_col1:
    html_link = st.text_input("🌐 Klistra in HTML-länk till rapport (valfritt):", key="html_link_input")
    uploaded_file = st.file_uploader("📎 Eller ladda upp rapportfil:",
        type=["pdf", "txt", "html", "docx", "md", "png", "jpg", "jpeg"], key="file_uploader_input")
with input_col2:
    manual_text_input = st.text_area("✏️ Eller klistra in text manuellt här (valfritt):", height=205, key="manual_text_input_area")

preview_text, ocr_extracted_text = "", ""

if uploaded_file:
    if uploaded_file.name.endswith((".png", ".jpg", ".jpeg")):
        ocr_extracted_text, _ = extract_text_from_image_or_pdf(uploaded_file)
        if ocr_extracted_text: st.expander("🖼️ OCR-utläst text (förhandsvisning)").text(ocr_extracted_text[:2000])
        else: st.warning("Kunde inte extrahera text med OCR från bilden.")
    else:
        preview_text = extract_text_from_file(uploaded_file)
elif html_link:
    preview_text = fetch_html_text(html_link)

text_to_analyze = manual_text_input or preview_text or ocr_extracted_text

if text_to_analyze:
    st.expander("📜 Text som kommer att analyseras (förhandsvisning)", expanded=False).text(text_to_analyze[:3000])
else:
    st.info("ℹ️ Ange en rapport via länk, filuppladdning eller inklistrad text för att kunna starta en analys.")

# --- Analysalternativ med Tabs ---
st.header("⚙️ Välj Analysmetod")
tab_full_analysis, tab_rag_analysis = st.tabs(["🔍 Fullständig Rapportanalys", "💬 Ställ en Fråga till Rapporten"])

with tab_full_analysis:
    if st.button("Starta fullständig analys", use_container_width=True, key="btn_full_analysis_tab_main"):
        if text_to_analyze and len(text_to_analyze.strip()) > 20:
            with st.spinner("📊 GPT analyserar hela rapporten..."):
                ai_report_content = full_rapportanalys(text_to_analyze)
                st.session_state['ai_report_content'] = ai_report_content
                st.markdown("### 🧾 Fullständig AI-analys:")
                st.markdown(st.session_state['ai_report_content'])
        else:
            st.error("Ingen text tillgänglig för fullständig analys, eller texten är för kort.")

    if 'ai_report_content' in st.session_state and st.session_state['ai_report_content']:
        st.download_button(label="Ladda ner fullständig analys som PDF",
                           data=answer_to_pdf(st.session_state['ai_report_content']),
                           file_name="ai_full_analys.pdf",
                           mime="application/pdf",
                           use_container_width=True,
                           key="dl_full_report_pdf_tab_main")

with tab_rag_analysis:
    if "user_question_rag_tab" not in st.session_state:
        st.session_state.user_question_rag_tab = "Vilken utdelning per aktie föreslås för nästa år?"
    st.text_input("Din specifika fråga om rapporten:", key="user_question_rag_tab")

    if st.button("Starta frågebaserad analys", key="btn_rag_analysis_tab_main", use_container_width=True):
        if text_to_analyze and len(text_to_analyze.strip()) > 20:
            with st.spinner("🤖 GPT söker och analyserar baserat på din fråga..."):
                source_id = (html_link or (uploaded_file.name if uploaded_file else text_to_analyze[:50])) + "_embeddings_v5"
                cache_file = get_embedding_cache_name(source_id)
                embedded_chunks = load_embeddings_if_exists(cache_file)

                if not embedded_chunks:
                    st.info("Skapar och cachar text-embeddings (kan ta en stund för stora dokument)...")
                    chunks = chunk_text(text_to_analyze)
                    embedded_chunks = []
                    if chunks:
                        progress_bar = st.progress(0, text="Bearbetar textblock...")
                        for i, chunk_content in enumerate(chunks, 1):
                            try:
                                embedding = get_embedding(chunk_content)
                                embedded_chunks.append({"text": chunk_content, "embedding": embedding})
                                progress_bar.progress(i / len(chunks), text=f"Bearbetar textblock {i}/{len(chunks)}")
                            except Exception as e_emb:
                                st.error(f"❌ Fel vid embedding av chunk {i}: {e_emb}")
                                st.stop()
                        save_embeddings(cache_file, embedded_chunks)
                        progress_bar.empty()
                        st.success("Embeddings skapade och cachade!")
                    else:
                        st.warning("Kunde inte skapa några textblock (chunks) från den angivna texten.")
                        st.stop()

                if not embedded_chunks:
                    st.error("Inga embeddings tillgängliga för analys.")
                    st.stop()

                retrieved_context, top_chunks_details = search_relevant_chunks(
                    st.session_state.user_question_rag_tab, embedded_chunks
                )

                st.expander("Relevant kontext som skickas till GPT").code(retrieved_context[:2000], language="text")

                # Ingen mer extra prompt-information från aktiedata här
                final_question_for_rag = st.session_state.user_question_rag_tab
                
                # Kan tas bort eller behållas om du vill visa frågan som skickas till GPT
                # st.expander("Slutgiltig fråga som skickas till GPT").caption(final_question_for_rag)

                rag_answer_content = generate_gpt_answer(final_question_for_rag, retrieved_context)
                st.session_state['rag_answer_content'] = rag_answer_content

                st.markdown(f"### 🤖 GPT-svar:\n{rag_answer_content}")

                if rag_answer_content:
                    st.markdown("--- \n ### Automatisk AI-evaluering (RAGAS):")
                    ragas_result = ragas_evaluate(
                        st.session_state.user_question_rag_tab,
                        rag_answer_content,
                        [chunk_text_content for _, chunk_text_content in top_chunks_details]
                    )
                    if ragas_result and "error" in ragas_result:
                        st.info(f"(RAGAS) Kunde inte utvärdera svaret: {ragas_result['error']}")
                    elif ragas_result:
                        faith_score = ragas_result.get('faithfulness')
                        ans_rel_score = ragas_result.get('answer_relevancy')
                        col_ragas1, col_ragas2 = st.columns(2)
                        with col_ragas1:
                            st.metric("Faithfulness", f"{faith_score:.2f}" if faith_score is not None else "N/A",
                                      help="Mäter hur väl AI:ns svar grundar sig på den information som hämtats från rapporten (0-1). Högre är bättre.")
                        with col_ragas2:
                            st.metric("Answer Relevancy", f"{ans_rel_score:.2f}" if ans_rel_score is not None else "N/A",
                                      help="Mäter hur relevant AI:ns svar är på den ställda frågan (0-1). Högre är bättre.")
        else:
            st.error("Ingen text tillgänglig för frågebaserad analys, eller så är texten för kort.")

    if 'rag_answer_content' in st.session_state and st.session_state['rag_answer_content']:
        st.markdown("---")
        st.subheader("⬇️ Exportera GPT-frågesvar")
        col_export_rag1, col_export_rag2 = st.columns(2)
        with col_export_rag1:
            st.download_button(
                "💾 Ladda ner svar (.txt)",
                st.session_state['rag_answer_content'],
                file_name="gpt_frågesvar.txt",
                key="dl_gpt_txt_rag_tab_main",
                use_container_width=True
            )
        with col_export_rag2:
            st.download_button(
                "📄 Ladda ner svar (.pdf)",
                answer_to_pdf(st.session_state['rag_answer_content']),
                file_name="gpt_frågesvar.pdf",
                key="dl_gpt_pdf_rag_tab_main",
                use_container_width=True
            )
