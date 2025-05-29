# 📊 AI-Rapportanalys

**AI-baserad analys av årsrapporter och företagsdokument – med GPT-4o, RAG och embeddings-cache**

---

## 🚀 Funktioner

- 🔍 Analys av PDF, HTML, textfiler och bilder (med OCR)
- 🧠 Frågebaserad sökning med GPT-4o och Retrieval-Augmented Generation (RAG)
- 📊 Automatisk identifiering av nyckeltal, utdelning, resultat, risker mm
- 📤 Exportera AI-svar som PDF eller txt
- 💾 Embeddings-cache för snabba och billiga återanalyser
- 🗂️ Tydlig och modulär kodstruktur
- 🌍 Stöd för svenska och engelska rapporter

---

## 🖥️ Teknik & bibliotek

- **Python 3.10+**
- [Streamlit](https://streamlit.io)  
- [OpenAI API (GPT-4o)](https://platform.openai.com/)
- PyMuPDF, BeautifulSoup, scikit-learn, dotenv, Tesseract/EasyOCR m.fl.

---

## 📦 Mappstruktur

ai-rapportanalys/
├── app.py # Streamlit-huvudfil
├── core/ # GPT-logik, chunking, embeddings
├── data/ # Cache, exporter, uppladdningar
│ ├── embeddings/
│ ├── outputs/
│ └── uploads/
├── services/ # API-klienter, HTML-nedladdning
├── utils/ # Hjälpmoduler: OCR, PDF-export, filutils m.m.
├── requirements.txt # Lista över alla Python-paket
├── README.md # (Du är här!)
└── .gitignore # Exkluderar känsliga/tempfiler från Git

yaml
Kopiera
Redigera

---

## ▶️ Så kör du projektet lokalt

1. **Klon repo:**
   ```bash
   git clone https://github.com/siffror/ai-rapportanalys.git
   cd ai-rapportanalys
Installera beroenden:

bash
Kopiera
Redigera
pip install -r requirements.txt
Lägg till OpenAI API-nyckel:
Skapa en fil .env i root-mappen:

ini
Kopiera
Redigera
OPENAI_API_KEY=ditt-api-nyckel-här
Starta appen:

bash
Kopiera
Redigera
streamlit run app.py
🧠 Hur funkar det?
Ladda upp eller länka till rapport (PDF, HTML, TXT, bild)

Texten extraheras och delas upp i “chunks”

Embeddings skapas och cachas lokalt

Du ställer en fråga – appen hittar de mest relevanta textbitarna

GPT-4o besvarar frågan – endast utifrån rapportens innehåll!

Exportera svaret som PDF eller txt vid behov

🌐 Demo & länk
👉 Testa live på Streamlit Cloud

🔐 Tips om API-nyckel
Dela aldrig din API-nyckel i koden eller på GitHub!

Använd .env lokalt och Streamlit Cloud “Secrets” vid deployment.

✨ Kontakt & credits
Utvecklat av @Siffror Zakaria
Som del av utbildningen DS24 Data Science

Välkommen att lämna feedback eller skapa “issues” om du hittar buggar eller vill bidra!
