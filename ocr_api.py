from fastapi import FastAPI, UploadFile
from .ocr_app import deepseek_ocr  # your OCR function

app = FastAPI()


@app.post("/api/ocr")
async def perfom_ocr(
    file: UploadFile,
    rotation: int = Form(0),
    mode: str = Form("Raw text"),
):
    # save temp file
    tmp = tempfile.NamedTemporaryFile(delete=False)
    shutil.copyfileobj(file.file, tmp)
    tmp.close()

    text = run_ocr(tmp.name, rotation, mode)

    return {"text": text}