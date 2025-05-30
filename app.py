import streamlit as st
from dotenv import load_dotenv
import os
import requests # För Lottie och yahooquery session

# RAGAS och LLM funktioner
from utils.evaluation_utils import ragas_evaluate # Se till att denna är anpassad för din RAGAS-version
from core.gpt_logic import (
    search_relevant_chunks, 
    generate_gpt_answer,
    chunk_text, 
    full_rapportanalys
)
# Importera get_embedding från där den är definierad (t.ex. core.embedding_utils eller core.gpt_logic)
from core.embedding_utils import get_embedding 

# Fil- och datahantering
from core.file_processing import extract_text_from_file
from utils.cache_utils import get_embedding_cache_name, save_embeddings, load_embeddings_if_exists
from utils.ocr_utils import extract_text_from_image_or_pdf
from utils.pdf_utils import answer_to_pdf
from utils.file_utils import save_output_file, save_uploaded_file
from services.html_downloader import fetch_html_text

# För aktiedata och tickersökning
from yahooquery import Ticker, search as yahoo_ticker_search
from streamlit_lottie import st_lottie

# --- Skapa nödvändiga datamappar ---
for d in ["data/embeddings", "data/outputs", "data/uploads"]:
    os.makedirs(d, exist_ok=True)

load_dotenv()
st.set_page_config(page_title="🤖 AI Rapportanalys", layout="wide")

# --- Funktioner för Aktiedata ---
@st.cache_data(ttl=3600) # Cache i 1 timme
def get_stock_metrics(ticker_symbol: str):
    st.write(f"[Debug] Hämtar nyckeltal för: {ticker_symbol}")
    if not ticker_symbol:
        return {"error": "Ticker saknas."}
    try:
        # Använd en session för yahooquery för potentiellt bättre stabilitet/prestanda
        session = requests.Session()
        t = Ticker(ticker_symbol, validate=True, progress=False, session=session)
        
        price_data = t.price.get(ticker_symbol, {}) if hasattr(t, 'price') and t.price else {}
        current_price = price_data.get('regularMarketPrice')
        currency = price_data.get('currency')
        market_cap = price_data.get('marketCap')

        summary_detail_data = t.summary_detail.get(ticker_symbol, {}) if hasattr(t, 'summary_detail') and t.summary_detail else {}
        pe_ratio = summary_detail_data.get('trailingPE')
        dividend_yield_raw = summary_detail_data.get('dividendYield')
        dividend_rate_annual = summary_detail_data.get('dividendRate')
        fifty_two_week_high = summary_detail_data.get('fiftyTwoWeekHigh')
        fifty_two_week_low = summary_detail_data.get('fiftyTwoWeekLow')

        key_stats_data = t.key_stats.get(ticker_symbol, {}) if hasattr(t, 'key_stats') and t.key_stats else {}
        beta = key_stats_data.get('beta')

        if market_cap is None:
            summary_profile_data = t.summary_profile.get(ticker_symbol, {}) if hasattr(t, 'summary_profile') and t.summary_profile else {}
            market_cap = summary_profile_data.get('marketCap') 
        if market_cap is None:
            financial_data = t.financial_data.get(ticker_symbol, {}) if hasattr(t, 'financial_data') and t.financial_data else {}
            market_cap = financial_data.get('marketCap')

        return {
            'Price': current_price, 'Currency': currency, 'MarketCap': market_cap,
            'PE': pe_ratio, 'Beta': beta,
            'DirectYield': dividend_yield_raw * 100 if dividend_yield_raw is not None else None,
            'Dividend': dividend_rate_annual, '52WeekHigh': fifty_two_week_high,
            '52WeekLow': fifty_two_week_low,
        }
    except Exception as e:
        st.error(f"Ett oväntat fel inträffade vid hämtning av aktiedata för {ticker_symbol}: {e}")
        return {"error": f"Kunde inte hämta data för {ticker_symbol}."}

@st.cache_data(ttl=3600)
def search_for_tickers_by_name(company_query: str) -> list:
    if not company_query:
        return []
    try:
        st.write(f"[Debug] Söker efter tickers för företagsnamn: {company_query}")
        search_results = yahoo_ticker_search(company_query)
        quotes_found = search_results.get('quotes', [])
        
        valid_results = []
        for quote in quotes_found:
            if isinstance(quote, dict) and 'symbol' in quote and ('shortname' in quote or 'longname' in quote):
                # Exempel på filter: Inkludera bara aktier (EQUITY)
                if quote.get('quoteType') == 'EQUITY':
                    valid_results.append(quote)
        st.write(f"[Debug] Hittade {len(valid_results)} giltiga aktie-tickers.")
        return valid_results
    except Exception as e:
        st.error(f"Fel vid tickersökning för '{company_query}': {e}")
        return []

# --- UI Start ---

# Lottie-animation
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
    except requests.exceptions.JSONDecodeError as e_json: # Korrigerat feltyp
        st.warning(f"Kunde inte tolka AI-animationen (JSON-fel): {e_json}")

st.markdown("<h1 style='color:#3EA6FF;'>🤖 AI-baserad Rapportanalys</h1>", unsafe_allow_html=True)

# --- Sidebar för Tickersökning och Aktieinformation ---
with st.sidebar:
    st.header("🔍 Sök Aktie")
    company_name_search_query = st.text_input(
        "Sök företagsnamn för att hitta ticker:", 
        key="company_search_input_key" # Unik nyckel
    )

    # Session state för den valda/inmatade tickern
    if 'selected_ticker_for_metrics' not in st.session_state:
        st.session_state.selected_ticker_for_metrics = "VOLV-B.ST" # Defaultvärde

    if company_name_search_query: # Om användaren har skrivit något i sökfältet
        found_tickers = search_for_tickers_by_name(company_name_search_query)
        if found_tickers:
            ticker_options_display = {"": "Välj en aktie från sökresultat..."} # Tomt förstaval
            for item in found_tickers:
                name = item.get('shortname', item.get('longname', 'Okänt namn'))
                symbol = item['symbol']
                exchange_display = item.get('exchDisp', item.get('exchange', 'Okänd börs'))
                display_str = f"{name} ({symbol}) - {exchange_display}"
                ticker_options_display[display_str] = symbol
            
            selected_display_str = st.selectbox(
                "Sökresultat:",
                options=list(ticker_options_display.keys()),
                key="search_result_selectbox_key", # Unik nyckel
                index=0 # Default till det tomma valet
            )
            if selected_display_str and ticker_options_display[selected_display_str] != "": # Om ett giltigt val gjorts
                st.session_state.selected_ticker_for_metrics = ticker_options_display[selected_display_str]
                # Återställ sökfältet så att selectboxen försvinner om man vill söka igen (valfritt)
                # st.session_state.company_search_input_key = "" # Kan orsaka omedelbar omkörning
        elif company_name_search_query:
            st.info("Inga aktier hittades för din sökning.")

    st.header("📈 Aktieinformation")
    # Ticker-inmatningsfältet använder och uppdaterar session state
    st.text_input(
        "Aktieticker:", 
        key="selected_ticker_for_metrics" # Kopplad till session_state
    )

    stock_metrics_data = None
    if st.session_state.selected_ticker_for_metrics:
        stock_metrics_data = get_stock_metrics(st.session_state.selected_ticker_for_metrics)

    if stock_metrics_data:
        if "error" in stock_metrics_data:
            st.warning(f"Kunde inte hämta marknadsdata för {st.session_state.selected_ticker_for_metrics}: {stock_metrics_data['error']}")
        else:
            ticker_display_name = st.session_state.selected_ticker_for_metrics.upper()
            currency_val = stock_metrics_data.get('Currency', '')
            st.markdown(f"#### {ticker_display_name} ({currency_val})")
            
            # Visa nyckeltal
            price_val = stock_metrics_data.get('Price')
            st.metric("Senaste pris", f"{price_val:.2f} {currency_val}" if price_val is not None else "–")
            market_cap_val = stock_metrics_data.get('MarketCap')
            if market_cap_val is not None:
                if market_cap_val >= 1e9: market_cap_display = f"{market_cap_val/1e9:.2f} Mdr {currency_val}"
                elif market_cap_val >= 1e6: market_cap_display = f"{market_cap_val/1e6:.2f} Mkr {currency_val}"
                else: market_cap_display = f"{market_cap_val:,.0f} {currency_val}"
                st.metric("Marknadsvärde", market_cap_display)
            else: st.metric("Marknadsvärde", "–")
            
            pe_val = stock_metrics_data.get('PE')
            st.metric("P/E-tal", f"{pe_val:.2f}" if pe_val is not None else "–")
            beta_val = stock_metrics_data.get('Beta')
            st.metric("Beta", f"{beta_val:.2f}" if beta_val is not None else "–")
            direct_yield_val = stock_metrics_data.get('DirectYield')
            st.metric("Direktavkastning", f"{direct_yield_val:.2f} %" if direct_yield_val is not None else "–")
            dividend_val = stock_metrics_data.get('Dividend')
            st.metric("Årlig utdelning/aktie", f"{dividend_val:.2f} {currency_val}" if dividend_val is not None else "–")
            high_52w_val = stock_metrics_data.get('52WeekHigh')
            st.metric("52 v. hög", f"{high_52w_val:.2f} {currency_val}" if high_52w_val is not None else "–")
            low_52w_val = stock_metrics_data.get('52WeekLow')
            st.metric("52 v. låg", f"{low_52w_val:.2f} {currency_val}" if low_52w_val is not None else "–")

# --- Huvudinnehåll: Input för rapportanalys ---
st.header("📄 Mata in rapporttext")
html_link = st.text_input("🌐 Klistra in HTML-länk till rapport (valfritt):")
uploaded_file = st.file_uploader("📎 Eller ladda upp rapportfil (PDF, TXT, HTML, DOCX etc.):", 
    type=["html", "pdf", "txt", "docx", "md", "png", "jpg", "jpeg"]) # Utökat filtyper lite
manual_text_input = st.text_area("✏️ Eller klistra in text manuellt här (valfritt):", "", height=200)

preview_text, ocr_extracted_text = "", ""

if uploaded_file:
    # save_path = save_uploaded_file(uploaded_file) # Spara bara om nödvändigt, eller hantera temporärt
    # st.info(f"Uppladdad fil: {uploaded_file.name}") # Mindre info om serverväg
    if uploaded_file.name.endswith((".png", ".jpg", ".jpeg")):
        ocr_extracted_text, _ = extract_text_from_image_or_pdf(uploaded_file)
        if ocr_extracted_text: st.text_area("🖼️ OCR-utläst text:", ocr_extracted_text[:2000], height=150)
        else: st.warning("Kunde inte extrahera text med OCR från bilden.")
    else:
        preview_text = extract_text_from_file(uploaded_file)
elif html_link:
    preview_text = fetch_html_text(html_link)

# Om manuell text finns och ingen fil/länk, använd den. Annars prioriteras fil/länk.
text_to_analyze = preview_text or ocr_extracted_text or manual_text_input

if text_to_analyze:
    st.text_area("📜 Text som kommer att analyseras (förhandsvisning):", text_to_analyze[:3000], height=150)
else:
    st.info("ℹ️ Ange en rapport via länk, filuppladdning eller inklistrad text för att kunna starta en analys.")

# --- Analysalternativ ---
st.header("⚙️ Välj Analysmetod")
col_analyze1, col_analyze2 = st.columns(2)

with col_analyze1:
    if st.button("🔍 Generera fullständig rapportanalys", use_container_width=True):
        if text_to_analyze and len(text_to_analyze.strip()) > 20:
            with st.spinner("📊 GPT analyserar hela rapporten..."):
                st.markdown("### 🧾 Fullständig AI-analys:")
                ai_report_content = full_rapportanalys(text_to_analyze) # Antag att denna tar hänsyn till språk etc.
                st.session_state['ai_report_content'] = ai_report_content
                st.markdown(ai_report_content) # Visa rapporten
        else:
            st.error("Ingen text tillgänglig för fullständig analys, eller texten är för kort.")

    if 'ai_report_content' in st.session_state and st.session_state['ai_report_content']:
        if st.button("📄 Spara fullständig AI-analys som PDF", key="save_full_report_pdf_main", use_container_width=True):
            pdf_bytes = answer_to_pdf(st.session_state['ai_report_content'])
            # output_path = save_output_file("ai_full_analys.pdf", pdf_bytes) # Spara på server om det är avsikten
            # st.success(f"PDF för fullständig analys har sparats till servern: {output_path}")
            st.download_button(label="Ladda ner fullständig analys som PDF",
                               data=pdf_bytes,
                               file_name="ai_full_analys.pdf",
                               mime="application/pdf",
                               use_container_width=True)


with col_analyze2:
    if "user_question_rag" not in st.session_state: # Nyckel för RAG-fråga
        st.session_state.user_question_rag = "Vilken utdelning per aktie föreslås för nästa år?"
    st.text_input("Din specifika fråga om rapporten:", key="user_question_rag")

    if st.button("💬 Analysera med GPT baserat på fråga", key="analyze_with_rag", use_container_width=True):
        if text_to_analyze and len(text_to_analyze.strip()) > 20:
            with st.spinner("🤖 GPT söker och analyserar baserat på din fråga..."):
                # Embedding och chunk-logik
                source_id = (html_link or (uploaded_file.name if uploaded_file else text_to_analyze[:50])) + "_embeddings_v3"
                cache_file = get_embedding_cache_name(source_id)
                embedded_chunks = load_embeddings_if_exists(cache_file)

                if not embedded_chunks:
                    st.info("Skapar och cachar text-embeddings (kan ta en stund för stora dokument)...")
                    chunks = chunk_text(text_to_analyze)
                    embedded_chunks = []
                    if chunks:
                        progress_bar = st.progress(0, text="Bearbetar textblock...")
                        for i, chunk_content in enumerate(chunks, 1): # Byt namn på variabel
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
                    st.session_state.user_question_rag, embedded_chunks
                )
                
                st.expander("Relevant kontext som skickas till GPT").code(retrieved_context[:2000], language="text")

                extra_prompt_for_rag = ""
                if stock_metrics_data and "error" not in stock_metrics_data:
                    # Bygg prompten med tillgängliga värden, hantera None
                    beta_val = stock_metrics_data.get('Beta')
                    pe_val = stock_metrics_data.get('PE')
                    yield_val = stock_metrics_data.get('DirectYield')
                    beta_str = f"{beta_val:.2f}" if isinstance(beta_val, (int, float)) else str(beta_val if beta_val is not None else "N/A")
                    pe_str = f"{pe_val:.2f}" if isinstance(pe_val, (int, float)) else str(pe_val if pe_val is not None else "N/A")
                    yield_str = f"{yield_val:.2f}%" if isinstance(yield_val, (int, float)) else str(yield_val if yield_val is not None else "N/A")
                    extra_prompt_for_rag = (
                        f"Ta hänsyn till följande aktuella marknadsdata för bolaget när du formulerar ditt svar: "
                        f"Beta={beta_str}, P/E-tal={pe_str}, Direktavkastning={yield_str}.\n"
                    )
                
                final_question_for_rag = extra_prompt_for_rag + st.session_state.user_question_rag
                if extra_prompt_for_rag: # Visa bara om marknadsdata lades till
                     st.expander("Slutgiltig fråga som skickas till GPT (inkl. marknadsdata)").caption(final_question_for_rag)


                rag_answer_content = generate_gpt_answer(final_question_for_rag, retrieved_context)
                st.session_state['rag_answer_content'] = rag_answer_content
                
                st.success("✅ Svar på fråga klart!")
                st.markdown(f"### 🤖 GPT-svar:\n{rag_answer_content}")

                if rag_answer_content:
                    st.markdown("--- \n ### Automatisk AI-evaluering (RAGAS):")
                    # Använd ursprunglig fråga för relevansbedömning, inte den med marknadsdata
                    ragas_result = ragas_evaluate(
                        st.session_state.user_question_rag, 
                        rag_answer_content,
                        [chunk_text_content for _, chunk_text_content in top_chunks_details]
                    )
                    if ragas_result and "error" in ragas_result: # Kontrollera att ragas_result inte är None
                        st.info(f"(RAGAS) Kunde inte utvärdera svaret: {ragas_result['error']}")
                    elif ragas_result: # Kontrollera att ragas_result inte är None
                        faith_score = ragas_result.get('faithfulness')
                        ans_rel_score = ragas_result.get('answer_relevancy')
                        col_ragas1, col_ragas2 = st.columns(2)
                        with col_ragas1:
                            st.metric("Faithfulness", f"{faith_score:.2f}" if faith_score is not None else "N/A",
                                      help="Mäter hur väl AI:ns svar grundar sig på den information som hämtats från rapporten. Högre är bättre.")
                        with col_ragas2:
                            st.metric("Answer Relevancy", f"{ans_rel_score:.2f}" if ans_rel_score is not None else "N/A",
                                      help="Mäter hur relevant AI:ns svar är på den ställda frågan. Högre är bättre.")
        else:
            st.error("Ingen text tillgänglig för frågebaserad analys, eller så är texten för kort.")

    if 'rag_answer_content' in st.session_state and st.session_state['rag_answer_content']:
        st.markdown("---") # Avdelare
        # Flyttat exportknappar till att vara direkt under RAG-svaret, men utanför with col_analyze2
        # för bättre layout om de tar plats. Eller behåll dem i kolumnen om det ser bättre ut.
        
# Knappar för att ladda ner och spara frågebaserat svar (visas om det finns i session state)
if 'rag_answer_content' in st.session_state and st.session_state['rag_answer_content']:
    st.subheader("⬇️ Exportera GPT-frågesvar")
    col_export1, col_export2 = st.columns(2) # Kanske bara två exportknappar här
    with col_export1:
        st.download_button(
            "💾 Ladda ner svar (.txt)",
            st.session_state['rag_answer_content'],
            file_name="gpt_frågesvar.txt",
            key="dl_gpt_txt_rag_main",
            use_container_width=True
        )
    with col_export2:
        st.download_button(
            "📄 Ladda ner svar (.pdf)",
            answer_to_pdf(st.session_state['rag_answer_content']),
            file_name="gpt_frågesvar.pdf",
            key="dl_gpt_pdf_rag_main",
            use_container_width=True
        )
    # Spara till server-knapp kan vara här eller tas bort om den inte används ofta
    # if st.button("📤 Spara GPT-frågesvar som PDF på servern", key="save_rag_answer_pdf_server_main", use_container_width=True):
    #     pdf_bytes = answer_to_pdf(st.session_state['rag_answer_content'])
    #     output_path = save_output_file("gpt_frågesvar_server.pdf", pdf_bytes)
    #     st.success(f"PDF för frågesvar har sparats till servern: {output_path}")
