import os

def save_output_file(filename: str, data: bytes):
    output_dir = os.path.join("data", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "wb") as f:
        f.write(data)
    return output_path
