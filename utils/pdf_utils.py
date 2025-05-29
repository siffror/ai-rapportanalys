
from fpdf import FPDF

def answer_to_pdf(answer: str) -> bytes:
    """
    Konverterar ett GPT-svar till en PDF och returnerar det som bytes.

    Args:
        answer (str): Textbaserat svar från GPT.

    Returns:
        bytes: PDF-data redo att laddas ner.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    for line in answer.split("\n"):
        pdf.multi_cell(0, 10, line)

    return pdf.output(dest="S").encode("latin1")
