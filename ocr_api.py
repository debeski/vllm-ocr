from fastapi import FastAPI, UploadFile, File, Form
import shutil
from pathlib import Path
import tempfile

app = FastAPI()

@app.post("/api/ocr")
async def ocr_endpoint(
    file: UploadFile = File(...),
    rotation: int = Form(0),
    mode: str = Form("Raw text")
):
    # ALWAYS use the original filename
    original_name = file.filename  
    suffix = Path(original_name).suffix.lower()

    # Save uploaded file exactly as-is with its extension preserved
    tmp_dir = tempfile.gettempdir()
    saved_path = Path(tmp_dir) / original_name

    with open(saved_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Wrap into an object matching validate_file expectations
    class FileWrapper:
        def __init__(self, path):
            self.name = str(path)

    wrapped_file = FileWrapper(saved_path)

    from ocr_app import run_ocr  # import OCR function

    text = run_ocr(wrapped_file, rotation, mode)

    return {"text": text}
