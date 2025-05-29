import pdfplumber
from bs4 import BeautifulSoup
import pandas as pd
import streamlit as st
from utils.ocr_utils import extract_text_from_image_or_pdf

def extract_text_from_file(file):
    text_output = ""
    if file.name.endswith(".pdf"):
        file.seek(0)
        try:
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_output += page_text + "\n"
        except Exception as e:
            st.warning(f"⚠️ Kunde inte läsa PDF: {e}")

    elif file.name.endswith(".html"):
        soup = BeautifulSoup(file.read(), "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text_output = soup.get_text(separator="\n")

    elif file.name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file)
        text_output = df.to_string(index=False)

    return text_output
