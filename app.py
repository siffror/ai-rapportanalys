# Importerar nödvändiga bibliotek och moduler
import streamlit as st  # För att bygga webbapplikationen
from dotenv import load_dotenv # För att ladda miljövariabler från en .env-fil
import os # För operativsysteminteraktioner, t.ex. filhantering
import requests # För att göra HTTP-anrop (används här för Lottie-animationen)

# Importerar anpassade funktioner för RAGAS (Retrieval Augmented Generation Assessment) och LLM (Large Language Model)
from utils.evaluation_utils import ragas_evaluate # Funktion för att utvärdera RAG-systemet
from core.gpt_logic import (
    search_relevant_chunks,    # Funktion för att hitta relevanta textdelar (chunks)
    generate_gpt_answer,       # Funktion för att generera svar med GPT
    chunk_text,                # Funktion för att dela upp text i mindre delar
    full_rapportanalys         # Funktion för att göra en fullständig rapportanalys
)
from core.embedding_utils import get_embedding # Funktion för att skapa text-embeddings

# Importerar anpassade funktioner för fil- och datahantering
from core.file_processing import extract_text_from_file # Funktion för att extrahera text från olika filtyper
from utils.cache_utils import get_embedding_cache_name, save_embeddings, load_embeddings_if_exists # Funktioner för att hantera cachning av embeddings
from utils.ocr_utils import extract_text_from_image_or_pdf # Funktion för att extrahera text från bilder eller PDFer med OCR
from utils.pdf_utils import answer_to_pdf # Funktion för att konvertera text till PDF
from utils.file_utils import save_output_file, save_uploaded_file # Funktioner för att spara filer
from services.html_downloader import fetch_html_text # Funktion för att hämta textinnehåll från en HTML-länk

# Importerar bibliotek för Lottie-animationen
from streamlit_lottie import st_lottie # Komponent för att visa Lottie-animationer i Streamlit
import traceback # För att få detaljerad felinformation vid undantag

# --- Skapa nödvändiga datamappar ---
# Ser till att kataloger för lagring av embeddings, output-filer och uppladdade filer existerar.
# exist_ok=True förhindrar fel om mapparna redan finns.
for d in ["data/embeddings", "data/outputs", "data/uploads"]:
    os.makedirs(d, exist_ok=True)

# Laddar miljövariabler från .env-filen (t.ex. API-nycklar)
load_dotenv()

# Konfigurerar Streamlit-sidans titel och layout
st.set_page_config(page_title="🤖 AI Rapportanalys", layout="wide")

# --- UI Start ---
# Skapar kolumner för att centrera Lottie-animationen
col1_lottie, col2_lottie, col3_lottie = st.columns([3, 4, 3])
with col2_lottie:
    # URL till Lottie-animationen
    lottie_url = "https://raw.githubusercontent.com/siffror/ai-rapportanalys/main/1GRWuk0lXN.json"
    try:
        # Försöker hämta Lottie-animationen från URL:en med en timeout
        r = requests.get(lottie_url, timeout=10)
        r.raise_for_status() # Genererar ett undantag om HTTP-anropet misslyckades (t.ex. 404, 500)
        lottie_json = r.json() # Tolkar svaret som JSON
        # Visar Lottie-animationen i Streamlit-appen
        st_lottie(lottie_json, speed=1, width=240, height=240, loop=True, quality="high", key="ai_logo")
    except requests.exceptions.RequestException as e_req:
        # Visar en varning om animationen inte kunde laddas p.g.a. nätverksfel
        st.warning(f"Kunde inte ladda AI-animationen (nätverksfel): {e_req}")
    except requests.exceptions.JSONDecodeError as e_json_lottie:
        # Visar en varning om animationen inte kunde tolkas som JSON
        st.warning(f"Kunde inte tolka AI-animationen (JSON-fel): {e_json_lottie}")

# Visar huvudtiteln för applikationen med anpassad HTML-styling
st.markdown("<h1 style='color:#3EA6FF;'>🤖 AI-baserad Rapportanalys</h1>", unsafe_allow_html=True)

# --- Sidomenyn är borttagen ---
# (Ingen kod här, men kommentaren indikerar att sidomenyn avsiktligt inte används)

# --- Huvudinnehåll: Input för rapportanalys ---
st.header("📄 Mata in rapporttext") # Rubrik för inmatningssektionen
# Skapar två kolumner för att organisera inmatningsfälten
input_col1, input_col2 = st.columns(2)

with input_col1:
    # Textinmatningsfält för HTML-länk
    html_link = st.text_input("🌐 Klistra in HTML-länk till rapport (valfritt):", key="html_link_input")
    # Filuppladdare för olika filtyper
    uploaded_file = st.file_uploader("📎 Eller ladda upp rapportfil:",
                                     type=["pdf", "txt", "html", "docx", "md", "png", "jpg", "jpeg"], key="file_uploader_input")
with input_col2:
    # Textområde för manuell inmatning av text
    manual_text_input = st.text_area("✏️ Eller klistra in text manuellt här (valfritt):", height=205, key="manual_text_input_area")

# Initierar variabler för text som ska analyseras
preview_text, ocr_extracted_text = "", ""

# Logik för att hantera den uppladdade filen
if uploaded_file:
    # Kontrollerar om filen är en bild (för OCR)
    if uploaded_file.name.endswith((".png", ".jpg", ".jpeg")):
        # Extraherar text från bilden med OCR
        ocr_extracted_text, _ = extract_text_from_image_or_pdf(uploaded_file)
        if ocr_extracted_text:
            # Visar en förhandsgranskning av den OCR-extraherade texten (max 2000 tecken)
            st.expander("🖼️ OCR-utläst text (förhandsvisning)").text(ocr_extracted_text[:2000])
        else:
            st.warning("Kunde inte extrahera text med OCR från bilden.")
    else:
        # Extraherar text från andra filtyper (PDF, TXT, DOCX etc.)
        preview_text = extract_text_from_file(uploaded_file)
elif html_link:
    # Hämtar textinnehåll från den angivna HTML-länken
    preview_text = fetch_html_text(html_link)

# Bestämmer vilken text som ska användas för analysen baserat på användarens input
# Prioriteringsordning: manuell inmatning, sedan text från uppladdad fil/HTML-länk (inklusive OCR)
text_to_analyze = manual_text_input or preview_text or ocr_extracted_text

# Visar en förhandsgranskning av texten som kommer att analyseras
if text_to_analyze:
    st.expander("📜 Text som kommer att analyseras (förhandsvisning)", expanded=False).text(text_to_analyze[:3000])
else:
    # Informerar användaren om att input behövs för att starta analysen
    st.info("ℹ️ Ange en rapport via länk, filuppladdning eller inklistrad text för att kunna starta en analys.")

# --- Analysalternativ med Tabs ---
st.header("⚙️ Välj Analysmetod") # Rubrik för val av analysmetod
# Skapar två flikar: en för fullständig rapportanalys och en för frågebaserad analys (RAG)
tab_full_analysis, tab_rag_analysis = st.tabs(["🔍 Fullständig Rapportanalys", "💬 Ställ en Fråga till Rapporten"])

# Innehåll för fliken "Fullständig Rapportanalys"
with tab_full_analysis:
    # Knapp för att starta den fullständiga analysen
    if st.button("Starta fullständig analys", use_container_width=True, key="btn_full_analysis_tab_main"):
        # Kontrollerar om det finns tillräckligt med text för analys
        if text_to_analyze and len(text_to_analyze.strip()) > 20:
            # Visar en spinner medan GPT analyserar rapporten
            with st.spinner("📊 GPT analyserar hela rapporten..."):
                # Anropar funktionen för fullständig rapportanalys
                ai_report_content = full_rapportanalys(text_to_analyze)
                # Sparar resultatet i session state för att kunna återanvändas (t.ex. för nedladdning)
                st.session_state['ai_report_content'] = ai_report_content
                # Visar rubrik och den genererade AI-analysen
                st.markdown("### 🧾 Fullständig AI-analys:")
                st.markdown(st.session_state['ai_report_content'])
        else:
            # Visar ett felmeddelande om ingen text finns eller om texten är för kort
            st.error("Ingen text tillgänglig för fullständig analys, eller texten är för kort.")

    # Om en fullständig analys har genererats och finns i session state
    if 'ai_report_content' in st.session_state and st.session_state['ai_report_content']:
        # Visar en nedladdningsknapp för att spara analysen som PDF
        st.download_button(label="Ladda ner fullständig analys som PDF",
                            data=answer_to_pdf(st.session_state['ai_report_content']), # Konverterar analysen till PDF-format
                            file_name="ai_full_analys.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            key="dl_full_report_pdf_tab_main")

# Innehåll för fliken "Ställ en Fråga till Rapporten" (RAG-analys)
with tab_rag_analysis:
    # Initierar en exempelfråga i session state om den inte redan finns
    if "user_question_rag_tab" not in st.session_state:
        st.session_state.user_question_rag_tab = "Vilken utdelning per aktie föreslås för nästa år?"
    # Textinmatningsfält för användarens specifika fråga
    st.text_input("Din specifika fråga om rapporten:", key="user_question_rag_tab")

    # Knapp för att starta den frågebaserade analysen
    if st.button("Starta frågebaserad analys", key="btn_rag_analysis_tab_main", use_container_width=True):
        # Kontrollerar om det finns tillräckligt med text för analys
        if text_to_analyze and len(text_to_analyze.strip()) > 20:
            # Visar en spinner medan GPT söker och analyserar
            with st.spinner("🤖 GPT söker och analyserar baserat på din fråga..."):
                # Skapar en unik identifierare för källan (används för cachning av embeddings)
                source_id = (html_link or (uploaded_file.name if uploaded_file else text_to_analyze[:50])) + "_embeddings_v5"
                # Hämtar namnet på cache-filen för embeddings
                cache_file = get_embedding_cache_name(source_id)
                # Försöker ladda förberäknade embeddings från cache
                embedded_chunks = load_embeddings_if_exists(cache_file)

                # Om inga cachade embeddings hittades
                if not embedded_chunks:
                    st.info("Skapar och cachar text-embeddings (kan ta en stund för stora dokument)...")
                    # Delar upp texten i mindre "chunks"
                    chunks = chunk_text(text_to_analyze)
                    embedded_chunks = [] # Lista för att lagra textdelar och deras embeddings
                    if chunks:
                        # Visar en progress bar för bearbetningen av textblock
                        progress_bar = st.progress(0, text="Bearbetar textblock...")
                        for i, chunk_content in enumerate(chunks, 1):
                            try:
                                # Skapar embedding för varje textblock
                                embedding = get_embedding(chunk_content)
                                embedded_chunks.append({"text": chunk_content, "embedding": embedding})
                                # Uppdaterar progress bar
                                progress_bar.progress(i / len(chunks), text=f"Bearbetar textblock {i}/{len(chunks)}")
                            except Exception as e_emb:
                                # Hanterar fel som kan uppstå vid skapande av embeddings
                                st.error(f"❌ Fel vid embedding av chunk {i}: {e_emb}")
                                st.stop() # Avbryter körningen vid fel
                        # Sparar de nyskapade embeddings till cache-filen
                        save_embeddings(cache_file, embedded_chunks)
                        progress_bar.empty() # Tar bort progress bar
                        st.success("Embeddings skapade och cachade!")
                    else:
                        # Varnar om inga textblock kunde skapas
                        st.warning("Kunde inte skapa några textblock (chunks) från den angivna texten.")
                        st.stop() # Avbryter körningen

                # Om inga embeddings finns (antingen från cache eller nyskapade)
                if not embedded_chunks:
                    st.error("Inga embeddings tillgängliga för analys.")
                    st.stop() # Avbryter körningen

                # Söker efter de mest relevanta textblocken baserat på användarens fråga och de skapade embeddings
                retrieved_context, top_chunks_details = search_relevant_chunks(
                    st.session_state.user_question_rag_tab, embedded_chunks
                )

                # Visar den relevanta kontexten som kommer att skickas till GPT (max 2000 tecken)
                st.expander("Relevant kontext som skickas till GPT").code(retrieved_context[:2000], language="text")

                # Den slutgiltiga frågan som skickas till GPT är användarens fråga.
                # Ingen ytterligare prompt-modifiering sker här i denna version.
                final_question_for_rag = st.session_state.user_question_rag_tab
                
                # Kommentar: Denna expander kan användas för att visa den exakta frågan som skickas till GPT.
                # st.expander("Slutgiltig fråga som skickas till GPT").caption(final_question_for_rag)

                # Genererar ett svar från GPT baserat på frågan och den hämtade kontexten
                rag_answer_content = generate_gpt_answer(final_question_for_rag, retrieved_context)
                # Sparar RAG-svaret i session state
                st.session_state['rag_answer_content'] = rag_answer_content

                # Visar GPT:s svar
                st.markdown(f"### 🤖 GPT-svar:\n{rag_answer_content}")

                # Om ett RAG-svar har genererats
                if rag_answer_content:
                    st.markdown("--- \n ### Automatisk AI-evaluering (RAGAS):") # Rubrik för RAGAS-utvärdering
                    # Utvärderar kvaliteten på RAG-svaret med RAGAS
                    ragas_result = ragas_evaluate(
                        st.session_state.user_question_rag_tab, # Användarens fråga
                        rag_answer_content,                     # GPT:s svar
                        [chunk_text_content for _, chunk_text_content in top_chunks_details] # Textinnehållet från de relevanta chunks
                    )
                    # Hanterar resultatet från RAGAS-utvärderingen
                    if ragas_result and "error" in ragas_result:
                        st.info(f"(RAGAS) Kunde inte utvärdera svaret: {ragas_result['error']}")
                    elif ragas_result:
                        # Hämtar RAGAS-metrics: Faithfulness och Answer Relevancy
                        faith_score = ragas_result.get('faithfulness')
                        ans_rel_score = ragas_result.get('answer_relevancy')
                        # Visar RAGAS-metrics i två kolumner
                        col_ragas1, col_ragas2 = st.columns(2)
                        with col_ragas1:
                            st.metric("Faithfulness", f"{faith_score:.2f}" if faith_score is not None else "N/A",
                                      help="Mäter hur väl AI:ns svar grundar sig på den information som hämtats från rapporten (0-1). Högre är bättre.")
                        with col_ragas2:
                            st.metric("Answer Relevancy", f"{ans_rel_score:.2f}" if ans_rel_score is not None else "N/A",
                                      help="Mäter hur relevant AI:ns svar är på den ställda frågan (0-1). Högre är bättre.")
        else:
            # Visar ett felmeddelande om ingen text finns eller om texten är för kort för RAG-analys
            st.error("Ingen text tillgänglig för frågebaserad analys, eller så är texten för kort.")

    # Om ett RAG-svar finns i session state
    if 'rag_answer_content' in st.session_state and st.session_state['rag_answer_content']:
        st.markdown("---") # Horisontell linje
        st.subheader("⬇️ Exportera GPT-frågesvar") # Rubrik för exportalternativ
        # Skapar två kolumner för nedladdningsknappar
        col_export_rag1, col_export_rag2 = st.columns(2)
        with col_export_rag1:
            # Knapp för att ladda ner RAG-svaret som en .txt-fil
            st.download_button(
                "💾 Ladda ner svar (.txt)",
                st.session_state['rag_answer_content'],
                file_name="gpt_frågesvar.txt",
                key="dl_gpt_txt_rag_tab_main",
                use_container_width=True
            )
        with col_export_rag2:
            # Knapp för att ladda ner RAG-svaret som en .pdf-fil
            st.download_button(
                "📄 Ladda ner svar (.pdf)",
                answer_to_pdf(st.session_state['rag_answer_content']), # Konverterar svaret till PDF
                file_name="gpt_frågesvar.pdf",
                key="dl_gpt_pdf_rag_tab_main",
                use_container_width=True
            )
