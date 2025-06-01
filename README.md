[![Streamlit Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-rapport-analys-ds24.streamlit.app/)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-8e44ad?logo=openai&logoColor=white)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)


---

# 📊 **AI-Rapportanalys**

**AI-baserad analys av årsrapporter och företagsdokument – med GPT-4o, RAG och embeddings-cache**

---

## 🌐 **Demo & länk**

[👉Testa live på Streamlit Cloud](https://ai-rapportanalys-ds24.streamlit.app/)

---

## 🚀 **Funktioner**

- 🔍 Analys av PDF, HTML, textfiler och bilder (med OCR)
- 🧠 Frågebaserad sökning med GPT-4o och Retrieval-Augmented Generation (RAG)
- 📊 Automatisk identifiering av nyckeltal, utdelning, resultat, risker mm
- 📤 Exportera AI-svar som PDF eller txt
- 💾 Embeddings-cache för snabba och billiga återanalyser
- 🗂️ Tydlig och modulär kodstruktur
- 🌍 Stöd för svenska och engelska rapporter

---

## 🖥️ **Teknik & bibliotek**

- **Python 3.10+**
- [Streamlit](https://streamlit.io)  
- [OpenAI API (GPT-4o)](https://platform.openai.com/)
- PyMuPDF, BeautifulSoup, scikit-learn, dotenv, Tesseract/EasyOCR m.fl.

---

```
## 📦 Mappstruktur


ai-rapportanalys/
├── app.py                 # Streamlit-huvudfil (appens gränssnitt)
├── gpt_server.py          # (ev. separat serverdel för GPT-logik)
├── core/                  # GPT-logik, chunking, embedding, filhantering
│   ├── __init__.py
│   ├── chunking.py
│   ├── embedding_utils.py
│   ├── file_processing.py
│   └── gpt_logic.py
├── data/                  # Data och resultat
│   ├── outputs/
│   └── uploads/
├── services/              # API-klienter, HTML-nedladdning
│   ├── html_downloader.py
│   └── openai_service.py
├── utils/                 # Hjälpmoduler: OCR, PDF, cache, mm
│   ├── __init__.py
│   ├── cache_utils.py
│   ├── evaluation_utils.py
│   ├── file_utils.py
│   ├── general.py
│   ├── ocr_utils.py
│   └── pdf_utils.py
├── requirements.txt       # Lista över Python-paket
├── README.md              # Dokumentation
├── .gitignore             # Exkluderar tempfiler från Git
└── 1GRWuk0IXN.json        # (logo)


```
---

## ▶️ **Så kör du projektet lokalt**

1. **Klon repo:**
   ```bash
   git clone https://github.com/siffror/ai-rapportanalys.git
   cd ai-rapportanalys
   
Installera beroenden:
   ```bash

pip install -r requirements.txt
```
Lägg till OpenAI API-nyckel:
Skapa en fil .env i root-mappen:
   ```bash

OPENAI_API_KEY=ditt-api-nyckel-här
```
Starta appen:
   ```bash
streamlit run app.py
```


## 🧠 **Hur funkar det?**

Ladda upp eller länka till rapport (PDF, HTML, TXT, bild)

Texten extraheras och delas upp i “chunks”

Embeddings skapas och cachas lokalt

Du ställer en fråga – appen hittar de mest relevanta textbitarna

GPT-4o besvarar frågan – endast utifrån rapportens innehåll!

Exportera svaret som PDF eller txt vid behov


## 🔐 **Tips om API-nyckel**

Dela aldrig din API-nyckel i koden eller på GitHub!

Använd .env lokalt och Streamlit Cloud “Secrets” vid deployment.

## ✨ **Kontakt & credits**

Utvecklat av [@siffror](https://github.com/siffror)  
LinkedIn: [zakariyae-Mokhtari](https://www.linkedin.com/in/zakariyae-mokhtari/)  

[![siffror](https://github.com/siffror.png?size=50)](https://github.com/siffror)


Som del av utbildningen DS24 Data Science
Välkommen att lämna feedback eller skapa “issues” om du hittar buggar eller vill bidra!
