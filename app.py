import streamlit as st
from dotenv import load_dotenv
from utils.evaluation_utils import ragas_evaluate
from yahooquery import Ticker

from core.gpt_logic import (
    search_relevant_chunks, generate_gpt_answer, # get_embedding (används indirekt via search_relevant_chunks)
    chunk_text # full_rapportanalys (importeras men används inte i den här versionen, se nedan)
)
# Om full_rapportanalys faktiskt ska användas, se till att den importeras och anropas korrekt.
# Just nu finns det kod för "Fullständig rapportanalys" som anropar en funktion med det namnet.
from core.gpt_logic import full_rapportanalys # Se till att denna är korrekt

from core.file_processing import extract_text_from_file
from utils.cache_utils import get_embedding_cache_name, save_embeddings, load_embeddings_if_exists
# from utils.general import is_key_figure # Oanvänd, kan tas bort om den inte behövs
from utils.ocr_utils import extract_text_from_image_or_pdf
from utils.pdf_utils import answer_to_pdf
from utils.file_utils import save_output_file, save_uploaded_file
from services.html_downloader import fetch_html_text

import os
import requests
from streamlit_lottie import st_lottie

# --- Funktion för att hämta nyckeltal (med cachning) ---
@st.cache_data(ttl=3600) # Cache i 1 timme
def get_stock_metrics(ticker_symbol: str):
    st.write(f"[Debug] Hämtar nyckeltal för: {ticker_symbol}") # För att se när den körs
    if not ticker_symbol:
        return {"error": "Ticker saknas."}
    try:
        t = Ticker(ticker_symbol)
        # Använd .get(ticker_symbol, {}) för att undvika KeyError om ticker inte finns i responsen
        key_stats_data = t.key_stats.get(ticker_symbol, {}) if t.key_stats else {}
        summary_detail_data = t.summary_detail.get(ticker_symbol, {}) if t.summary_detail else {}
        
        # Få fram PE-tal, Beta och direktavkastning
        pe = summary_detail_data.get('trailingPE')
        beta = key_stats_data.get('beta') # Beta finns oftast i key_stats
        dividend_yield = summary_detail_data.get('dividendYield')

        return {
            'PE': pe,
            'Beta': beta,
            'DirectYield': dividend_yield * 100 if dividend_yield is not None else None # Kontrollera None
        }
    except Exception as e:
        st.error(f"Fel vid hämtning av data för {ticker_symbol}: {e}")
        return {"error": str(e)}

# --- Skapa nödvändiga datamappar om de inte finns ---
for d in ["data/embeddings", "data/outputs", "data/uploads"]:
    os.makedirs(d, exist_ok=True)

load_dotenv()
st.set_page_config(page_title="🤖 AI Rapportanalys", layout="wide")

# --- Lottie-animation ---
# (Din Lottie-kod här - ingen ändring behövs)
col1_lottie, col2_lottie, col3_lottie = st.columns([3, 4, 3]) # Undvik namnkonflikt med senare kolumner
with col2_lottie:
    lottie_url = "https://raw.githubusercontent.com/siffror/ai-rapportanalys/main/1GRWuk0lXN.json"
    try:
        r = requests.get(lottie_url, timeout=10)
        r.raise_for_status() # Kollar om anropet lyckades (status code 2xx)
        lottie_json = r.json()
        st_lottie(lottie_json, speed=1, width=240, height=240, loop=True, quality="high", key="ai_logo")
    except requests.exceptions.RequestException as e:
        st.warning(f"Kunde inte ladda AI-animationen: {e}")
    except Exception as e_json: # Om json() misslyckas
         st.warning(f"Kunde inte tolka AI-animationen som JSON: {e_json}")


st.markdown("<h1 style='color:#3EA6FF;'>🤖 AI-baserad Rapportanalys</h1>", unsafe_allow_html=True)

# --- Input: HTML-länk, uppladdning, eller manuell text ---
html_link = st.text_input("🌐 Rapport-länk (HTML)")
uploaded_file = st.file_uploader("📎 Ladda upp HTML, PDF, bild eller text",
    type=["html", "pdf", "txt", "xlsx", "xls", "png", "jpg", "jpeg"])

preview_text, ocr_extracted_text = "", "" # Byt namn för tydlighet

# --- Hantera filuppladdning och textutvinning ---
if uploaded_file:
    save_path = save_uploaded_file(uploaded_file)
    st.info(f"Uppladdad fil har sparats som: {save_path}")
    if uploaded_file.name.endswith((".png", ".jpg", ".jpeg")):
        ocr_extracted_text, _ = extract_text_from_image_or_pdf(uploaded_file) # Använd ocr_extracted_text
        if ocr_extracted_text:
            st.text_area("📄 OCR-utläst text (förhandsvisning):", ocr_extracted_text[:2000], height=200)
        else:
            st.warning("Kunde inte extrahera text med OCR från bilden.")
    else:
        preview_text = extract_text_from_file(uploaded_file) # Använd preview_text
elif html_link:
    preview_text = fetch_html_text(html_link) # Använd preview_text
else:
    preview_text = st.text_area("✏️ Klistra in text manuellt här:", "", height=200) # Använd preview_text

# Kombinera extraherad text
text_to_analyze = preview_text or ocr_extracted_text

if text_to_analyze: # Visa bara om det finns någon text
    st.text_area("📄 Förhandsvisning av text som kommer att analyseras:", text_to_analyze[:5000], height=200)
else:
    st.warning("❌ Ingen text att analysera än. Ladda upp en fil, klistra in text eller ange en HTML-länk.")


# --- Aktieinformation och Nyckeltal ---
st.sidebar.header("📈 Aktieinformation")
ticker_symbol_input = st.sidebar.text_input("Aktieticker (t.ex. VOLV-B.ST):", value="VOLV-B.ST", key="ticker_input")
stock_metrics_data = None
if ticker_symbol_input:
    stock_metrics_data = get_stock_metrics(ticker_symbol_input) # Anropar cachad funktion

if stock_metrics_data:
    if "error" in stock_metrics_data:
        st.sidebar.warning(f"Kunde inte hämta marknadsdata: {stock_metrics_data['error']}")
    else:
        st.sidebar.markdown("### Marknadsdata / Nyckeltal")
        st.sidebar.metric("Beta", f"{stock_metrics_data['Beta']:.2f}" if stock_metrics_data['Beta'] is not None else "–")
        st.sidebar.metric("P/E-tal", f"{stock_metrics_data['PE']:.2f}" if stock_metrics_data['PE'] is not None else "–")
        st.sidebar.metric("Direktavkastning", f"{stock_metrics_data['DirectYield']:.2f} %" if stock_metrics_data['DirectYield'] is not None else "–")


# --- Fullständig rapportanalys ---
st.header("Analysalternativ")
if st.button("🔍 Generera fullständig rapportanalys"):
    if text_to_analyze and len(text_to_analyze.strip()) > 20:
        with st.spinner("📊 GPT analyserar hela rapporten..."):
            st.markdown("### 🧾 Fullständig AI-analys:")
            # Se till att 'full_rapportanalys' är korrekt definierad och importerad från core.gpt_logic
            ai_report_content = full_rapportanalys(text_to_analyze) # Nytt variabelnamn
            st.session_state['ai_report_content'] = ai_report_content # Spara i session state
            st.markdown(ai_report_content)
    else:
        st.error("Ingen text tillgänglig för fullständig analys.")

# Knappen för att spara fullständig analys visas om rapporten finns i session state
if 'ai_report_content' in st.session_state and st.session_state['ai_report_content']:
    if st.button("📄 Spara fullständig AI-analys som PDF", key="save_full_report_pdf"):
        pdf_bytes = answer_to_pdf(st.session_state['ai_report_content'])
        output_path = save_output_file("ai_full_analys.pdf", pdf_bytes)
        st.success(f"PDF för fullständig analys har sparats till servern: {output_path}")


# --- Frågebaserad GPT-analys ---
st.header("Ställ en specifik fråga")
if "user_question" not in st.session_state:
    st.session_state.user_question = "Vilken utdelning per aktie föreslås?"
st.text_input("Din fråga:", key="user_question")


if st.button("💬 Analysera med GPT baserat på fråga"): # Ändrat knapptext för tydlighet
    if text_to_analyze and len(text_to_analyze.strip()) > 20:
        with st.spinner("🤖 GPT analyserar baserat på din fråga..."):
            # Embedding och chunk-logik (din kod här - ingen ändring förutom variabelnamn)
            source_id = (html_link or uploaded_file.name if uploaded_file else text_to_analyze[:50]) + "-v2"
            cache_file = get_embedding_cache_name(source_id)
            embedded_chunks = load_embeddings_if_exists(cache_file)

            if not embedded_chunks:
                st.info("Skapar och cachar embeddings för dokumentet...")
                chunks = chunk_text(text_to_analyze)
                embedded_chunks = []
                progress_bar = st.progress(0)
                for i, chunk in enumerate(chunks, 1):
                    # st.write(f"🔹 Bearbetar chunk {i} av {len(chunks)}") # Kan vara för mycket output
                    try:
                        embedding = get_embedding(chunk) # get_embedding importeras från core.gpt_logic
                        embedded_chunks.append({"text": chunk, "embedding": embedding})
                        progress_bar.progress(i / len(chunks))
                    except Exception as e:
                        st.error(f"❌ Fel vid embedding av chunk {i}: {e}")
                        st.stop() # Stoppa om embedding misslyckas
                save_embeddings(cache_file, embedded_chunks)
                progress_bar.empty() # Ta bort progress bar när klar
                st.success("Embeddings skapade och cachade!")


            if not embedded_chunks:
                st.error("Inga embeddings tillgängliga för analys.")
                st.stop()

            # Hämta relevant kontext
            retrieved_context, top_chunks_details = search_relevant_chunks( # Byt namn på variabel
                st.session_state.user_question, embedded_chunks
            )
            
            st.subheader("Relevant kontext som skickas till GPT:")
            st.code(retrieved_context[:1500], language="text") # Visa lite mer kontext

            # --- Bygg prompt med marknadsdata (bara om stock_metrics_data finns!) ---
            extra_prompt_for_rag = ""
            if stock_metrics_data and "error" not in stock_metrics_data:
                # Se till att hämta värden säkert med .get()
                beta_val = stock_metrics_data.get('Beta', "N/A")
                pe_val = stock_metrics_data.get('PE', "N/A")
                yield_val = stock_metrics_data.get('DirectYield', "N/A")

                beta_str = f"{beta_val:.2f}" if isinstance(beta_val, (int, float)) else str(beta_val)
                pe_str = f"{pe_val:.2f}" if isinstance(pe_val, (int, float)) else str(pe_val)
                yield_str = f"{yield_val:.2f}%" if isinstance(yield_val, (int, float)) else str(yield_val)

                extra_prompt_for_rag = (
                    f"Ta hänsyn till följande marknadsdata för bolaget i ditt svar: Beta={beta_str}, "
                    f"P/E-tal={pe_str}, Direktavkastning={yield_str}.\n"
                )
            
            final_question_for_rag = extra_prompt_for_rag + st.session_state.user_question
            st.subheader("Slutgiltig fråga som skickas till GPT (inkl. marknadsdata):")
            st.caption(final_question_for_rag)

            # Generera svar baserat på fråga och kontext
            rag_answer_content = generate_gpt_answer(final_question_for_rag, retrieved_context) # Nytt variabelnamn
            st.session_state['rag_answer_content'] = rag_answer_content # Spara i session state
            
            st.success("✅ Svar klart!")
            st.markdown(f"### 🤖 GPT-4o svar:\n{rag_answer_content}")

            # --- RAGAS AI-evaluering ---
            if rag_answer_content: # Kör bara om det finns ett svar
                st.markdown("### Automatisk AI-evaluering (RAGAS):")
                # Använd ursprunglig fråga för relevansbedömning
                ragas_result = ragas_evaluate(
                    st.session_state.user_question, 
                    rag_answer_content,
                    [chunk_text_content for _, chunk_text_content in top_chunks_details] # Extrahera text från top_chunks
                )
                if "error" in ragas_result:
                    st.info(f"(RAGAS) Kunde inte utvärdera svaret: {ragas_result['error']}")
                else:
                    # Se till att nycklarna finns innan formatering
                    faith_score = ragas_result.get('faithfulness')
                    ans_rel_score = ragas_result.get('answer_relevancy')
                    col_ragas1, col_ragas2 = st.columns(2)
                    with col_ragas1:
                        st.metric("Faithfulness", f"{faith_score:.2f}" if faith_score is not None else "N/A")
                    with col_ragas2:
                        st.metric("Answer relevancy", f"{ans_rel_score:.2f}" if ans_rel_score is not None else "N/A")
    else:
        st.error("Ingen text tillgänglig för frågebaserad analys, eller så är texten för kort.")


# Knappar för att ladda ner och spara frågebaserat svar (visas om det finns i session state)
if 'rag_answer_content' in st.session_state and st.session_state['rag_answer_content']:
    st.subheader("Exportera frågesvar")
    col_export1, col_export2, col_export3 = st.columns(3)
    with col_export1:
        st.download_button(
            "💾 Ladda ner svar (.txt)",
            st.session_state['rag_answer_content'],
            file_name="gpt_frågesvar.txt",
            key="dl_gpt_txt_rag"
        )
    with col_export2:
        st.download_button(
            "📄 Ladda ner svar (.pdf)",
            answer_to_pdf(st.session_state['rag_answer_content']), # Skapa PDF från innehållet
            file_name="gpt_frågesvar.pdf",
            key="dl_gpt_pdf_rag"
        )
    with col_export3:
        if st.button("📤 Spara GPT-frågesvar som PDF på servern", key="save_rag_answer_pdf_server"):
            pdf_bytes = answer_to_pdf(st.session_state['rag_answer_content'])
            output_path = save_output_file("gpt_frågesvar_server.pdf", pdf_bytes)
            st.success(f"PDF för frågesvar har sparats till servern: {output_path}")


# Ta bort den felplacerade generate_gpt_answer och tillhörande prompt-logik härifrån.
# Den logiken är nu korrekt placerad inuti "💬 Analysera med GPT baserat på fråga"-knappen.
