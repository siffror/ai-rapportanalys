import requests
from bs4 import BeautifulSoup

def fetch_html_text(url: str) -> str:
    """
    Hämtar textinnehållet från en HTML-webbsida och rensar bort navigation, script, etc.
    """
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    return text
