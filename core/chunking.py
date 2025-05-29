# core/chunking.py

def chunk_text(text: str, max_length: int = 1500, overlap: int = 200) -> list:
    """
    Delar upp en lång text i mindre delar (chunks) med överlappning.

    Args:
        text (str): Originaltexten som ska delas upp.
        max_length (int): Maxlängd per chunk (standard: 1500 tecken).
        overlap (int): Antal tecken som överlappar mellan chunks (standard: 200).

    Returns:
        list: En lista med strängar (chunks).
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_length, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start += max_length - overlap
    return chunks
