# app.py (med utökad felsökning för sidopanelen)

import streamlit as st
from dotenv import load_dotenv
import os
import requests 

from utils.evaluation_utils import ragas_evaluate
from core.gpt_logic import (
    search_relevant_chunks, generate_gpt_answer,
    chunk_text, full_rapportanalys
)
from core.embedding_utils import get_embedding 
from core.file_processing import extract_text_from_file
from utils.cache_utils import get_embedding_cache_name, save_embeddings, load_embeddings_if_exists
from utils.ocr_utils import extract_text_from_image_or_pdf
from utils.pdf_utils import answer_to_pdf
from utils.file_utils import save_output_file, save_uploaded_file
from services.html_downloader import fetch_html_text

from yahooquery import Ticker, search as yahoo_ticker_search
from streamlit_lottie import st_lottie
import traceback

# --- Skapa nödvändiga datamappar ---
for d in ["data/embeddings", "data/outputs", "data/uploads"]:
    os.makedirs(d, exist_ok=True)

load_dotenv()
st.set_page_config(page_title="🤖 AI Rapportanalys", layout="wide")

# --- Funktioner för Aktiedata med MYCKET utökad DEBUG ---
@st.cache_data(ttl=3600) # Behåll cachning, men rensa cache på Streamlit Cloud vid behov under felsökning
def get_stock_metrics(ticker_symbol: str):
    st.write(f"--- [Debug get_stock_metrics V2] Startar för ticker: {ticker_symbol} ---")
    if not ticker_symbol:
        st.write("[Debug get_stock_metrics V2] Ticker är tom, returnerar fel.")
        return {"error": "Ticker saknas."}
    try:
        session = requests.Session()
        t = Ticker(ticker_symbol, validate=True, progress=False, session=session)
        st.write(f"[Debug get_stock_metrics V2] Ticker-objekt skapat: {t}")
        st.write(f"[Debug get_stock_metrics V2] Ticker-objektets moduler: {t.modules}")


        # Skriv ut rådata från varje modul FÖRE .get(ticker_symbol)
        price_module_content = t.price if hasattr(t, 'price') else None
        st.write(f"[Debug get_stock_metrics V2] Rådata från t.price (modulnivå): {price_module_content}")
        
        summary_detail_module_content = t.summary_detail if hasattr(t, 'summary_detail') else None
        st.write(f"[Debug get_stock_metrics V2] Rådata från t.summary_detail (modulnivå): {summary_detail_module_content}")

        key_stats_module_content = t.key_stats if hasattr(t, 'key_stats') else None
        st.write(f"[Debug get_stock_metrics V2] Rådata från t.key_stats (modulnivå): {key_stats_module_content}")
        
        summary_profile_module_content = t.summary_profile if hasattr(t, 'summary_profile') else None
        st.write(f"[Debug get_stock_metrics V2] Rådata från t.summary_profile (modulnivå): {summary_profile_module_content}")

        financial_data_module_content = t.financial_data if hasattr(t, 'financial_data') and t.financial_data else None
        st.write(f"[Debug get_stock_metrics V2] Rådata från t.financial_data (modulnivå): {financial_data_module_content}")

        # Försök hämta data för den specifika tickern från moduldatan
        price_data = price_module_content.get(ticker_symbol, {}) if isinstance(price_module_content, dict) else {}
        current_price = price_data.get('regularMarketPrice')
        currency = price_data.get('currency')
        market_cap = price_data.get('marketCap')

        summary_detail_data_dict = summary_detail_module_content.get(ticker_symbol, {}) if isinstance(summary_detail_module_content, dict) else {}
        pe_ratio = summary_detail_data_dict.get('trailingPE')
        dividend_yield_raw = summary_detail_data_dict.get('dividendYield')
        dividend_rate_annual = summary_detail_data_dict.get('dividendRate')
        fifty_two_week_high = summary_detail_data_dict.get('fiftyTwoWeekHigh')
        fifty_two_week_low = summary_detail_data_dict.get('fiftyTwoWeekLow')

        key_stats_data_dict = key_stats_module_content.get(ticker_symbol, {}) if isinstance(key_stats_module_content, dict) else {}
        beta = key_stats_data_dict.get('beta')

        if market_cap is None and isinstance(summary_profile_module_content, dict):
            summary_profile_data_dict = summary_profile_module_content.get(ticker_symbol, {})
            market_cap = summary_profile_data_dict.get('marketCap') 
        if market_cap is None and isinstance(financial_data_module_content, dict):
            financial_data_dict = financial_data_module_content.get(ticker_symbol, {})
            market_cap = financial_data_dict.get('marketCap')
        
        collected_metrics = {
            'Price': current_price, 'Currency': currency, 'MarketCap': market_cap,
            'PE': pe_ratio, 'Beta': beta,
            'DirectYield': dividend_yield_raw * 100 if dividend_yield_raw is not None else None,
            'Dividend': dividend_rate_annual, '52WeekHigh': fifty_two_week_high,
            '52WeekLow': fifty_two_week_low,
        }
        st.write(f"[Debug get_stock_metrics V2] Insamlade nyckeltal för {ticker_symbol}: {collected_metrics}")
        return collected_metrics
    except Exception as e:
        st.error(f"Fel i get_stock_metrics V2 för {ticker_symbol}: {type(e).__name__} - {e}")
        st.text_area(f"Traceback (get_stock_metrics V2 - {ticker_symbol}):", traceback.format_exc(), height=100, key=f"traceback_gsm_v2_{ticker_symbol}")
        return {"error": f"Kunde inte hämta data. Fel: {type(e).__name__}"}

@st.cache_data(ttl=3600)
def search_for_tickers_by_name(company_query: str) -> list:
    if not company_query:
        return []
    try:
        st.write(f"--- [Debug search_tickers V2] Startar sökning för: '{company_query}' ---")
        search_results = yahoo_ticker_search(company_query) # Använder alias
        st.write(f"[Debug search_tickers V2] Råa sökresultat från yahooquery: {str(search_results)[:1000]}...") # Begränsa utskrift

        quotes_found = search_results.get('quotes', [])
        st.write(f"[Debug search_tickers V2] Antal 'quotes' i rådata: {len(quotes_found)}")
        if quotes_found:
             st.write(f"[Debug search_tickers V2] Första 'quote'-objektet i rådata: {quotes_found[0]}")

        valid_results = []
        for i, quote in enumerate(quotes_found):
            # Mycket grundläggande kontroll för att se om det är en dictionary med en symbol
            if isinstance(quote, dict) and 'symbol' in quote:
                name_to_display = quote.get('shortname', quote.get('longname', 'Okänt namn'))
                st.write(f"[Debug search_tickers V2] Bearbetar quote {i+1}: Symbol={quote['symbol']}, Namn='{name_to_display}', Typ='{quote.get('quoteType')}'")
                # Behåll alla resultat för nu för att se vad vi får, oavsett quoteType
                valid_results.append(quote)
            else:
                st.write(f"[Debug search_tickers V2] Skippade quote {i+1} (inte dict eller saknar symbol): {quote}")
        
        st.write(f"[Debug search_tickers V2] Hittade {len(valid_results)} bearbetade tickers (innan ev. ytterligare filter).")
        return valid_results
    except Exception as e:
        st.error(f"Fel vid tickersökning V2 för '{company_query}': {type(e).__name__} - {e}")
        st.text_area(f"Traceback (search_tickers V2 - {company_query}):", traceback.format_exc(), height=100, key=f"traceback_search_v2_{company_query}")
        return []

# --- UI Start ---
# (Lottie-animation och titel som tidigare)
# ... (Din Lottie-kod och titel)

st.markdown("<h1 style='color:#3EA6FF;'>🤖 AI-baserad Rapportanalys</h1>", unsafe_allow_html=True) # Se till att detta är utanför Lottie-kolumnerna

# --- Sidebar för Tickersökning och Aktieinformation ---
with st.sidebar:
    st.header("🔍 Sök Aktie")
    company_name_search_query = st.text_input(
        "Sök företagsnamn för att hitta ticker:", 
        key="company_search_input_field_v2" # Nyckel för detta fält
    )

    if 'selected_ticker_for_metrics' not in st.session_state:
        st.session_state.selected_ticker_for_metrics = "VOLV-B.ST" # Defaultvärde

    if company_name_search_query: # Om användaren har skrivit något i sökfältet
        found_tickers = search_for_tickers_by_name(company_name_search_query) # Anropar den uppdaterade funktionen
        
        # Kontrollera om found_tickers faktiskt innehåller något
        if found_tickers:
            st.write(f"[Debug Sidebar] Antal found_tickers att skapa options från: {len(found_tickers)}")
            ticker_options_display = {"": "Välj en aktie från sökresultat..."} 
            for item in found_tickers:
                # Säkerställ att item är en dict och har nödvändiga nycklar
                if isinstance(item, dict) and 'symbol' in item:
                    name = item.get('shortname', item.get('longname', 'Okänt namn'))
                    symbol = item['symbol']
                    exchange_display = item.get('exchDisp', item.get('exchange', 'Okänd börs'))
                    quote_type_display = item.get('quoteType', 'Okänd typ')
                    display_str = f"{name} ({symbol}) - {exchange_display} [{quote_type_display}]"
                    ticker_options_display[display_str] = symbol
                else:
                    st.write(f"[Debug Sidebar] Skippade ogiltigt item vid skapande av options: {item}")
            
            st.write(f"[Debug Sidebar] ticker_options_display (antal efter placeholder): {len(ticker_options_display)-1}")

            if len(ticker_options_display) > 1: # Om vi har fler alternativ än bara placeholder
                def on_ticker_select_sidebar_v2():
                    selected_key_from_selectbox = st.session_state.search_result_selectbox_sidebar_key_v2
                    if selected_key_from_selectbox and ticker_options_display[selected_key_from_selectbox] != "":
                        st.session_state.selected_ticker_for_metrics = ticker_options_display[selected_key_from_selectbox]
                        st.session_state.company_search_input_field_v2 = "" 
                
                st.selectbox(
                    "Sökresultat:",
                    options=list(ticker_options_display.keys()),
                    key="search_result_selectbox_sidebar_key_v2", # Nyckel
                    index=0, # Default till det tomma valet
                    on_change=on_ticker_select_sidebar_v2
                )
            elif company_name_search_query: # Sökt, found_tickers fanns men blev inga options (bör inte hända med ny logik)
                 st.info("Inga formaterbara tickers hittades för din sökning.")
        elif company_name_search_query: # Sökt men found_tickers var tom från början
            st.info("Inga tickers returnerades från sökningen.")


    st.header("📈 Aktieinformation")
    st.text_input(
        "Aktieticker:", 
        key="selected_ticker_for_metrics" 
    )

    stock_metrics_data_display = None # Nytt variabelnamn för att undvika konflikt
    if st.session_state.selected_ticker_for_metrics:
        stock_metrics_data_display = get_stock_metrics(st.session_state.selected_ticker_for_metrics)

    if stock_metrics_data_display: # Använd det nya variabelnamnet
        if "error" in stock_metrics_data_display:
            st.warning(f"Kunde inte hämta marknadsdata för {st.session_state.selected_ticker_for_metrics}: {stock_metrics_data_display['error']}")
        else:
            # ... (Din kod för att visa st.metric, se till att den använder stock_metrics_data_display) ...
            # (Som i föregående "hela koden"-svar)
            ticker_display_name = st.session_state.selected_ticker_for_metrics.upper()
            currency_val = stock_metrics_data_display.get('Currency', '')
            st.markdown(f"#### {ticker_display_name} ({currency_val})")
            
            price_val = stock_metrics_data_display.get('Price')
            st.metric("Senaste pris", f"{price_val:.2f} {currency_val}" if price_val is not None else "–")
            # ... och så vidare för alla andra metriker, använd stock_metrics_data_display.get(...) ...


# --- Huvudinnehåll: Input för rapportanalys ---
# (Som i föregående "hela koden"-svar, med tabs etc.)
# ...

# --- Analysalternativ med Tabs ---
# (Som i föregående "hela koden"-svar)
# ...
# VIKTIGT: Se till att 'stock_metrics_data' som används i RAG-prompten (extra_prompt_for_rag)
# är samma som 'stock_metrics_data_display' ovan, eller att den hämtas på nytt baserat på
# st.session_state.selected_ticker_for_metrics om det behövs inom RAG-logiken.
# För enkelhetens skull kan du återanvända stock_metrics_data_display globalt efter att den definierats.
# Eller, inuti "Starta frågebaserad analys":
# current_ticker_for_rag = st.session_state.selected_ticker_for_metrics
# rag_stock_metrics = get_stock_metrics(current_ticker_for_rag) if current_ticker_for_rag else None
# if rag_stock_metrics and "error" not in rag_stock_metrics:
#     extra_prompt_for_rag = ... (bygg prompten)
