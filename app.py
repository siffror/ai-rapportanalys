import streamlit as st
from dotenv import load_dotenv
from utils.evaluation_utils import ragas_evaluate # Se till att denna är anpassad för din RAGAS-version
from yahooquery import Ticker

from core.gpt_logic import (
    search_relevant_chunks, generate_gpt_answer,
    chunk_text, full_rapportanalys
)
# get_embedding importeras vanligtvis inuti core.gpt_logic eller core.embedding_utils
# Om den används direkt här, se till att den är importerad.

from core.file_processing import extract_text_from_file
from utils.cache_utils import get_embedding_cache_name, save_embeddings, load_embeddings_if_exists
from utils.ocr_utils import extract_text_from_image_or_pdf
from utils.pdf_utils import answer_to_pdf
from utils.file_utils import save_output_file, save_uploaded_file
from services.html_downloader import fetch_html_text

import os
import requests
from streamlit_lottie import st_lottie

# --- Funktion för att hämta nyckeltal (med cachning och fler nyckeltal) ---
@st.cache_data(ttl=3600) # Cache i 1 timme
def get_stock_metrics(ticker_symbol: str):
    st.write(f"[Debug] Hämtar nyckeltal för: {ticker_symbol}") # För att se när den körs
    if not ticker_symbol:
        return {"error": "Ticker saknas."}
    try:
        t = Ticker(ticker_symbol, validate=True, progress=False, session=requests.Session()) # session kan hjälpa med stabilitet
        
        # Hämta all moduldata på en gång för effektivitet om möjligt
        # Eller hämta modul för modul och kontrollera om data finns
        
        price_data = t.price.get(ticker_symbol, {}) if hasattr(t, 'price') and t.price else {}
        current_price = price_data.get('regularMarketPrice')
        currency = price_data.get('currency')
        # MarketCap kan finnas i price_data eller summary_profile eller financial_data
        market_cap = price_data.get('marketCap')

        summary_detail_data = t.summary_detail.get(ticker_symbol, {}) if hasattr(t, 'summary_detail') and t.summary_detail else {}
        pe_ratio = summary_detail_data.get('trailingPE')
        dividend_yield_raw = summary_detail_data.get('dividendYield') # Detta är i decimalform
        dividend_rate_annual = summary_detail_data.get('dividendRate') # Årlig utdelning per aktie
        fifty_two_week_high = summary_detail_data.get('fiftyTwoWeekHigh')
        fifty_two_week_low = summary_detail_data.get('fiftyTwoWeekLow')

        key_stats_data = t.key_stats.get(ticker_symbol, {}) if hasattr(t, 'key_stats') and t.key_stats else {}
        beta = key_stats_data.get('beta')

        # Försök hämta MarketCap från andra källor om det saknas
        if market_cap is None: # Ofta i summaryProfile
            summary_profile_data = t.summary_profile.get(ticker_symbol, {}) if hasattr(t, 'summary_profile') and t.summary_profile else {}
            market_cap = summary_profile_data.get('marketCap') 
        if market_cap is None: # Kan också finnas i financialData
            financial_data = t.financial_data.get(ticker_symbol, {}) if hasattr(t, 'financial_data') and t.financial_data else {}
            market_cap = financial_data.get('marketCap')


        # Omsättning (Revenue) - Detta är mer komplicerat och kan kräva att man specificerar period (TTM, årlig, kvartalsvis)
        # revenue_ttm = financial_data.get('totalRevenue') # Ofta TTM (Trailing Twelve Months)
        # För nu utelämnar vi omsättning för att hålla det enklare, men det är en bra framtida utökning.

        return {
            'Price': current_price,
            'Currency': currency,
            'MarketCap': market_cap,
            'PE': pe_ratio,
            'Beta': beta,
            'DirectYield': dividend_yield_raw * 100 if dividend_yield_raw is not None else None,
            'Dividend': dividend_rate_annual,
            '52WeekHigh': fifty_two_week_high,
            '52WeekLow': fifty_two_week_low,
            # 'RevenueTTM': revenue_ttm, # Kan läggas till om du vill implementera det
        }
    except Exception as e:
        # Logga det faktiska felet för felsökning
        st.error(f"Ett oväntat fel inträffade vid hämtning av aktiedata för {ticker_symbol}: {e}")
        return {"error": f"Kunde inte hämta data för {ticker_symbol}. Försök igen eller kontrollera ticker."}


# --- Skapa nödvändiga datamappar om de inte finns ---
for d in ["data/embeddings", "data/outputs", "data/uploads"]:
    os.makedirs(d, exist_ok=True)

load_dotenv()
st.set_page_config(page_title="🤖 AI Rapportanalys", layout="wide")

# --- Lottie-animation ---
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
    except requests.exceptions.JSONDecodeError as e_json:
        st.warning(f"Kunde inte tolka AI-animationen (JSON-fel): {e_json}")


st.markdown("<h1 style='color:#3EA6FF;'>🤖 AI-baserad Rapportanalys</h1>", unsafe_allow_html=True)

# --- Input: HTML-länk, uppladdning, eller manuell text ---
html_link = st.text_input("🌐 Rapport-länk (HTML)")
uploaded_file = st.file_uploader("📎 Ladda upp HTML, PDF, bild eller text",
    type=["html", "pdf", "txt", "xlsx", "xls", "png", "jpg", "jpeg"])

preview_text, ocr_extracted_text = "", ""

if uploaded_file:
    save_path = save_uploaded_file(uploaded_file)
    st.info(f"Uppladdad fil har sparats som: {save_path}")
    if uploaded_file.name.endswith((".png", ".jpg", ".jpeg")):
        ocr_extracted_text, _ = extract_text_from_image_or_pdf(uploaded_file)
        if ocr_extracted_text:
            st.text_area("📄 OCR-utläst text (förhandsvisning):", ocr_extracted_text[:2000], height=200)
        else:
            st.warning("Kunde inte extrahera text med OCR från bilden.")
    else:
        preview_text = extract_text_from_file(uploaded_file)
elif html_link:
    preview_text = fetch_html_text(html_link)
else:
    preview_text = st.text_area("✏️ Klistra in text manuellt här:", "", height=200)

text_to_analyze = preview_text or ocr_extracted_text

if text_to_analyze:
    st.text_area("📄 Förhandsvisning av text som kommer att analyseras:", text_to_analyze[:5000], height=200)
else:
    st.warning("❌ Ingen text att analysera än. Ladda upp en fil, klistra in text eller ange en HTML-länk.")

# --- Aktieinformation och Nyckeltal i Sidebar ---
st.sidebar.header("📈 Aktieinformation")
ticker_symbol_input = st.sidebar.text_input("Aktieticker (t.ex. VOLV-B.ST eller AAPL):", value="VOLV-B.ST", key="ticker_input_sidebar")
stock_metrics_data = None

if ticker_symbol_input:
    stock_metrics_data = get_stock_metrics(ticker_symbol_input)

if stock_metrics_data:
    if "error" in stock_metrics_data:
        st.sidebar.warning(f"Kunde inte hämta marknadsdata för {ticker_symbol_input}: {stock_metrics_data['error']}")
    else:
        # Använd .get() med defaultvärde för att undvika KeyError om en nyckel saknas
        ticker_display_name = ticker_symbol_input.upper()
        currency_val = stock_metrics_data.get('Currency', '')
        st.sidebar.markdown(f"#### {ticker_display_name} ({currency_val})")
        
        price_val = stock_metrics_data.get('Price')
        st.sidebar.metric("Senaste pris", f"{price_val:.2f} {currency_val}" if price_val is not None else "–")

        market_cap_val = stock_metrics_data.get('MarketCap')
        if market_cap_val is not None:
            if market_cap_val >= 1e9: # Miljarder
                market_cap_display = f"{market_cap_val/1e9:.2f} Mdr {currency_val}"
            elif market_cap_val >= 1e6: # Miljoner
                market_cap_display = f"{market_cap_val/1e6:.2f} Mkr {currency_val}"
            else: # Mindre än en miljon
                market_cap_display = f"{market_cap_val:,.0f} {currency_val}" # Formatera med tusentalsavgränsare
            st.sidebar.metric("Marknadsvärde", market_cap_display)
        else:
            st.sidebar.metric("Marknadsvärde", "–")

        pe_val = stock_metrics_data.get('PE')
        st.sidebar.metric("P/E-tal", f"{pe_val:.2f}" if pe_val is not None else "–")
        
        beta_val = stock_metrics_data.get('Beta')
        st.sidebar.metric("Beta", f"{beta_val:.2f}" if beta_val is not None else "–")

        direct_yield_val = stock_metrics_data.get('DirectYield')
        st.sidebar.metric("Direktavkastning", f"{direct_yield_val:.2f} %" if direct_yield_val is not None else "–")

        dividend_val = stock_metrics_data.get('Dividend')
        st.sidebar.metric("Årlig utdelning/aktie", f"{dividend_val:.2f} {currency_val}" if dividend_val is not None else "–")

        high_52w_val = stock_metrics_data.get('52WeekHigh')
        st.sidebar.metric("52 v. hög", f"{high_52w_val:.2f} {currency_val}" if high_52w_val is not None else "–")
        
        low_52w_val = stock_metrics_data.get('52WeekLow')
        st.sidebar.metric("52 v. låg", f"{low_52w_val:.2f} {currency_val}" if low_52w_val is not None else "–")


# --- Fullständig rapportanalys ---
st.header("Analysalternativ")
if st.button("🔍 Generera fullständig rapportanalys"):
    if text_to_analyze and len(text_to_analyze.strip()) > 20:
        with st.spinner("📊 GPT analyserar hela rapporten..."):
            st.markdown("### 🧾 Fullständig AI-analys:")
            ai_report_content = full_rapportanalys(text_to_analyze)
            st.session_state['ai_report_content'] = ai_report_content
            st.markdown(ai_report_content)
    else:
        st.error("Ingen text tillgänglig för fullständig analys.")

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

if st.button("💬 Analysera med GPT baserat på fråga"):
    if text_to_analyze and len(text_to_analyze.strip()) > 20:
        with st.spinner("🤖 GPT analyserar baserat på din fråga..."):
            source_id = (html_link or (uploaded_file.name if uploaded_file else text_to_analyze[:50])) + "-v2_embeddings"
            cache_file = get_embedding_cache_name(source_id)
            embedded_chunks = load_embeddings_if_exists(cache_file)

            if not embedded_chunks:
                st.info("Skapar och cachar embeddings för dokumentet...")
                chunks = chunk_text(text_to_analyze)
                embedded_chunks = []
                if chunks: # Se till att chunks inte är tom
                    progress_bar = st.progress(0, text="Bearbetar textblock...")
                    for i, chunk in enumerate(chunks, 1):
                        try:
                            # Antag att get_embedding finns i core.gpt_logic eller core.embedding_utils
                            from core.embedding_utils import get_embedding # Importera här om den inte redan är globalt tillgänglig
                            embedding = get_embedding(chunk)
                            embedded_chunks.append({"text": chunk, "embedding": embedding})
                            progress_bar.progress(i / len(chunks), text=f"Bearbetar textblock {i}/{len(chunks)}")
                        except Exception as e:
                            st.error(f"❌ Fel vid embedding av chunk {i}: {e}")
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
                st.session_state.user_question, embedded_chunks
            )
            
            st.subheader("Relevant kontext som skickas till GPT:")
            st.code(retrieved_context[:1500], language="text")

            extra_prompt_for_rag = ""
            if stock_metrics_data and "error" not in stock_metrics_data:
                beta_val = stock_metrics_data.get('Beta')
                pe_val = stock_metrics_data.get('PE')
                yield_val = stock_metrics_data.get('DirectYield')
                beta_str = f"{beta_val:.2f}" if isinstance(beta_val, (int, float)) else str(beta_val if beta_val is not None else "N/A")
                pe_str = f"{pe_val:.2f}" if isinstance(pe_val, (int, float)) else str(pe_val if pe_val is not None else "N/A")
                yield_str = f"{yield_val:.2f}%" if isinstance(yield_val, (int, float)) else str(yield_val if yield_val is not None else "N/A")

                extra_prompt_for_rag = (
                    f"Ta hänsyn till följande aktuella marknadsdata för bolaget i ditt svar: Beta={beta_str}, "
                    f"P/E-tal={pe_str}, Direktavkastning={yield_str}.\n"
                )
            
            final_question_for_rag = extra_prompt_for_rag + st.session_state.user_question
            st.subheader("Slutgiltig fråga som skickas till GPT (inkl. marknadsdata om tillgängligt):")
            st.caption(final_question_for_rag)

            rag_answer_content = generate_gpt_answer(final_question_for_rag, retrieved_context)
            st.session_state['rag_answer_content'] = rag_answer_content
            
            st.success("✅ Svar klart!")
            st.markdown(f"### 🤖 GPT-4o svar:\n{rag_answer_content}")

            if rag_answer_content:
                st.markdown("### Automatisk AI-evaluering (RAGAS):")
                ragas_result = ragas_evaluate(
                    st.session_state.user_question, 
                    rag_answer_content,
                    [chunk_text_content for _, chunk_text_content in top_chunks_details]
                )
                if "error" in ragas_result:
                    st.info(f"(RAGAS) Kunde inte utvärdera svaret: {ragas_result['error']}")
                else:
                    faith_score = ragas_result.get('faithfulness')
                    ans_rel_score = ragas_result.get('answer_relevancy')
                    col_ragas1, col_ragas2 = st.columns(2)
                    with col_ragas1:
                        st.metric("Faithfulness", f"{faith_score:.2f}" if faith_score is not None else "N/A")
                    with col_ragas2:
                        st.metric("Answer relevancy", f"{ans_rel_score:.2f}" if ans_rel_score is not None else "N/A")
    else:
        st.error("Ingen text tillgänglig för frågebaserad analys, eller så är texten för kort.")

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
            answer_to_pdf(st.session_state['rag_answer_content']),
            file_name="gpt_frågesvar.pdf",
            key="dl_gpt_pdf_rag"
        )
    with col_export3:
        if st.button("📤 Spara GPT-frågesvar som PDF på servern", key="save_rag_answer_pdf_server"):
            pdf_bytes = answer_to_pdf(st.session_state['rag_answer_content'])
            output_path = save_output_file("gpt_frågesvar_server.pdf", pdf_bytes) # Byt namn för att undvika konflikt med full analys
            st.success(f"PDF för frågesvar har sparats till servern: {output_path}")
