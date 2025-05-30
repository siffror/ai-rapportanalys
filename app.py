import streamlit as st
from dotenv import load_dotenv
import os
import requests # För Lottie

# Hämta API-nycklar från secrets
alpha_key = st.secrets.get("ALPHAVANTAGE_API_KEY")
finnhub_key = st.secrets.get("FINNHUB_API_KEY")

# Imports för respektive API-klient
from alpha_vantage.timeseries import TimeSeries
import finnhub

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

# --- Nya funktioner för aktiepriser ---
@st.cache_data(ttl=1800)
def get_alpha_vantage_quote(symbol):
    if not alpha_key:
        return {"error": "Alpha Vantage API-nyckel saknas."}
    ts = TimeSeries(key=alpha_key, output_format='pandas')
    try:
        data, meta = ts.get_quote_endpoint(symbol=symbol)
        if data.empty or '05. price' not in data.columns:
             return {"error": f"Kunde inte hämta pris för {symbol} från Alpha Vantage. Kontrollera symbol eller API-kvot."}
        price = float(data['05. price'].iloc[0])
        # Du kan välja att returnera mer data från 'data' eller 'meta' om du vill visa det
        return {"price": price, "currency": meta.get("2. Symbol", symbol).split('.')[-1] if '.' in meta.get("2. Symbol", "") else "USD"} # Antagande om valuta, kan behöva justeras
    except Exception as e:
        st.error(f"Alpha Vantage fel för {symbol}: {e}")
        return {"error": str(e)}

@st.cache_data(ttl=1800)
def get_finnhub_quote(symbol):
    if not finnhub_key:
        return {"error": "Finnhub API-nyckel saknas."}
    client = finnhub.Client(api_key=finnhub_key)
    try:
        data = client.quote(symbol)
        price = data.get('c')
        if price is None or price == 0: # Finnhub kan returnera 0 för ogiltiga symboler eller när marknaden är stängd för vissa instrument
            return {"error": f"Kunde inte hämta aktuellt pris för {symbol} från Finnhub. Data: {data}"}
        # Finnhub returnerar inte valuta direkt i quote-endpointet, det måste hanteras separat
        # eller antas baserat på börsen om symbolen indikerar det (t.ex. ".ST" för SEK).
        currency_guess = "SEK" if symbol.upper().endswith(".ST") else "USD" # Mycket grov gissning
        return {"price": price, "currency": currency_guess, "data": data}
    except Exception as e:
        st.error(f"Finnhub fel för {symbol}: {e}")
        return {"error": str(e)}

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


# --- Sidebar för Aktieinformation (UPPDATERAD med Alpha Vantage & Finnhub) ---
with st.sidebar:
    st.header("📈 Aktiepris")

    # Behåll vald ticker i session_state (default: Volvo B)
    if 'selected_ticker_for_metrics' not in st.session_state:
        st.session_state.selected_ticker_for_metrics = "VOLV-B.ST" # Yahoo-format, kan behöva justeras för AV/Finnhub

    # Input för aktiesymbol
    st.session_state.selected_ticker_for_metrics = st.text_input(
        "Ange aktiesymbol:",
        value=st.session_state.selected_ticker_for_metrics,
        key="stock_symbol_input"
    ).upper() # Konvertera till versaler för konsistens

    col_av, col_fh = st.columns(2)
    fetch_av = False
    fetch_fh = False

    with col_av:
        if st.button("Pris (AlphaV)"):
            fetch_av = True
    with col_fh:
        if st.button("Pris (Finnhub)"):
            fetch_fh = True

    # Logik för att visa data
    # stock_metrics_data används fortfarande i RAG-flödet, men kommer nu bara innehålla pris.
    # Vi sätter det till None initialt för varje körning för att säkerställa att gammal data inte visas felaktigt.
    stock_metrics_data = None 
    # Håll kvar information om senaste framgångsrika hämtning för RAG
    if 'last_successful_stock_data' not in st.session_state:
        st.session_state.last_successful_stock_data = None


    if st.session_state.selected_ticker_for_metrics:
        api_source = None
        quote_data = None

        if fetch_av:
            api_source = "Alpha Vantage"
            with st.spinner(f"Hämtar från {api_source}..."):
                quote_data = get_alpha_vantage_quote(st.session_state.selected_ticker_for_metrics)
        elif fetch_fh:
            api_source = "Finnhub"
            with st.spinner(f"Hämtar från {api_source}..."):
                quote_data = get_finnhub_quote(st.session_state.selected_ticker_for_metrics)

        if quote_data:
            if "error" in quote_data:
                st.warning(f"Kunde inte hämta data för {st.session_state.selected_ticker_for_metrics} från {api_source}: {quote_data['error']}")
                stock_metrics_data = {"error": quote_data['error']} # För RAG-flödet
                st.session_state.last_successful_stock_data = {"error": quote_data['error']}
            else:
                price_val = quote_data.get('price')
                currency_val = quote_data.get('currency', '') # Hämta valuta om den finns
                st.markdown(f"#### {st.session_state.selected_ticker_for_metrics.upper()} ({api_source})")
                st.metric("Senaste pris", f"{price_val:.2f} {currency_val}" if price_val is not None else "–")
                
                # Uppdatera stock_metrics_data som RAG-flödet kan använda
                # Denna innehåller nu bara pris och valuta, inte de andra nyckeltalen.
                stock_metrics_data = {'Price': price_val, 'Currency': currency_val, 'Source': api_source}
                st.session_state.last_successful_stock_data = stock_metrics_data

                # Om du vill visa mer data från quote_data['data'], gör det här.
                # Exempel: st.json(quote_data['data'])
    elif fetch_av or fetch_fh:
        st.warning("Ange en aktiesymbol först.")
        
    # Se till att stock_metrics_data är tillgängligt för RAG även om ingen ny knapp tryckts denna körning
    if stock_metrics_data is None and st.session_state.last_successful_stock_data is not None:
        stock_metrics_data = st.session_state.last_successful_stock_data


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

                extra_prompt_for_rag = ""
                # Anpassa detta block om du vill skicka med prisinformationen till RAG-prompten
                # Använd 'stock_metrics_data' som nu innehåller pris och källa
                if stock_metrics_data and "error" not in stock_metrics_data:
                    price_info = stock_metrics_data.get('Price')
                    currency_info = stock_metrics_data.get('Currency', '')
                    source_info = stock_metrics_data.get('Source', 'okänd källa')
                    if price_info is not None:
                         extra_prompt_for_rag = (
                            f"Ta hänsyn till följande aktuella marknadsdata för bolaget när du formulerar ditt svar: "
                            f"Senaste pris: {price_info:.2f} {currency_info} (hämtat från {source_info}).\n"
                        )


                final_question_for_rag = extra_prompt_for_rag + st.session_state.user_question_rag_tab
                if extra_prompt_for_rag:
                     st.expander("Slutgiltig fråga som skickas till GPT (inkl. marknadsdata)").caption(final_question_for_rag)

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

