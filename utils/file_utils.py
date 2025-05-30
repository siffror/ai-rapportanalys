import os

def save_output_file(filename: str, data: bytes):
    output_dir = os.path.join("data", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "wb") as f:
        f.write(data)
    return output_path

def save_uploaded_file(uploaded_file):
    """
    Sparar en fil som laddats upp via Streamlit till data/uploads-mappen.
    """
    upload_dir = os.path.join("data", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    upload_path = os.path.join(upload_dir, uploaded_file.name)
    with open(upload_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return upload_path
