import gradio as gr
import torch
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
from PIL import Image
import fitz  # PyMuPDF
import logging
from pathlib import Path
import os
import re
import tempfile
import pandas as pd
from bs4 import BeautifulSoup

# ------------------ Configuration ------------------
os.environ["VLLM_COMPILE"] = "0"
os.environ["VLLM_TORCH_COMPILE_MODE"] = "none"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
vllm_logger = logging.getLogger("vllm")
vllm_logger.setLevel(logging.WARNING)

llm = None
pdf_pages_cache = {}

# ---------------- OCR Model Loader ----------------
def load_llm():
    """Initialize the LLM model once"""
    global llm
    if llm is not None:
        return
    
    # GPU check
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory
        gb = 1024 ** 3
        total_gb = total_mem / gb
        
        logger.info(f"Using GPU: {device_name}")
        logger.info(f"Available GPU Memory: {total_gb:.2f} GB")
    else:
        logger.warning("CUDA not available — using CPU.")

    # Load model
    llm = LLM(
        model="/workspace/model",
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        logits_processors=[NGramPerReqLogitsProcessor],
    )
    logger.info("OCR Model loaded successfully")

# ----------- OCR Text & Excel Functions -----------
def pdf_to_images(file_path):
    doc = fitz.open(file_path)
    images = []
    for page_num, page in enumerate(doc):
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images

def validate_file(file):
    if file is None:
        return None
    if not hasattr(file, 'name'):
        raise ValueError("Invalid file")
    path = Path(file.name)
    if not path.exists():
        raise ValueError(f"File not found: {file}")
    if path.stat().st_size > 20 * 1024**2:
        raise ValueError("File too large (max 20MB)")
    if path.suffix.lower() not in {'.pdf', '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tif', '.webp'}:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    return path

def run_ocr(file, rotation, mode):
    """Run OCR on uploaded file and return extracted text based on selected mode."""
    try:
        load_llm()
        file_path = validate_file(file)

        # Convert PDF to images
        if file_path.suffix.lower() == ".pdf":
            images = pdf_to_images(str(file_path))
        else:
            img = Image.open(str(file_path)).convert("RGB")
            images = [img]

        # Apply rotation (clockwise)
        rot = int(rotation)
        if rot != 0:
            images = [img.rotate(-rot, expand=True) for img in images]

        # Select prompt based on user choice
        if mode == "Raw text":
            prompt = "<image>\nFree OCR."
        else:
            prompt = "<image>\n<|grounding|>Convert the document to markdown."

        results = []
        for idx, img in enumerate(images):
            model_input = [{"prompt": prompt, "multi_modal_data": {"image": img}}]
            params = SamplingParams(
                temperature=0.0,
                max_tokens=8192,
                extra_args=dict(ngram_size=30, window_size=90, whitelist_token_ids={128821, 128822}),
                skip_special_tokens=False,
            )
            output = llm.generate(model_input, params)[0].outputs[0].text
            results.append(f"--- Page {idx + 1} ---\n{output}")

        return "\n\n".join(results)
    except Exception as e:
        logger.error(f"OCR error: {e}", exc_info=True)
        return f"❌ Error: {e}"

def extract_tables_to_excel(ocr_text, excel_path=None):
    """Extract <table> elements from OCR text to Excel file."""
    if not ocr_text.strip():
        return None

    # Use non-greedy matching to capture ALL tables
    table_blocks = re.findall(r"<table>(.*?)</table>", ocr_text, re.DOTALL)
    
    if not table_blocks:
        logger.info("No tables found in OCR text")
        return None

    logger.info(f"Found {len(table_blocks)} tables in OCR text")
    
    if excel_path is None:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".xlsx")
        os.close(tmp_fd)
        excel_path = tmp_path

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for idx, table_html in enumerate(table_blocks, start=1):
            try:
                soup = BeautifulSoup(f"<table>{table_html}</table>", "html.parser")
                rows = []
                for tr in soup.find_all("tr"):
                    row = [td.get_text(strip=True) for td in tr.find_all("td")]
                    if row:  # Only add non-empty rows
                        rows.append(row)
                if rows:  # Only create sheet if there are rows
                    df = pd.DataFrame(rows)
                    # Use shorter sheet name to avoid Excel limitations
                    sheet_name = f"Table_{idx}" if idx <= 10 else f"T{idx}"
                    df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
                    logger.info(f"Added table {idx} with {len(rows)} rows")
            except Exception as e:
                logger.error(f"Error processing table {idx}: {e}")
                continue

    return excel_path

def preview_loader(file, rotation):
    """Load all PDF pages or single image and return page 1 + total pages."""
    global pdf_pages_cache

    try:
        file_path = validate_file(file)

        # Convert to images
        if file_path.suffix.lower() == ".pdf":
            pages = pdf_to_images(str(file_path))
        else:
            pages = [Image.open(str(file_path)).convert("RGB")]

        # Apply rotation
        rot = int(rotation)
        if rot != 0:
            pages = [img.rotate(-rot, expand=True) for img in pages]

        # Store in cache
        pdf_pages_cache["pages"] = pages
        pdf_pages_cache["total"] = len(pages)

        # Return first page + total pages
        return pages[0], len(pages)
    except Exception as e:
        logger.error(f"Preview loader error: {e}", exc_info=True)
        return None, 0

def preview_page(page_number):
    """Return the selected page from cache."""
    try:
        pages = pdf_pages_cache.get("pages", [])
        total = pdf_pages_cache.get("total", 0)

        if not pages:
            return None

        # clamp between 1 and total
        page_number = max(1, min(int(page_number), total))

        return pages[page_number - 1]
    except Exception as e:
        logger.error(f"Page preview error: {e}")
        return None

def export_excel_from_text(text):
    """Gradio wrapper for exporting Excel."""
    path = extract_tables_to_excel(text)
    return path

# -------------- Gradio App Loader  ---------------
def launch_gradio():

    # Gradio UI
    with gr.Blocks() as demo:
        gr.Markdown("# DeepSeek-OCR نسخة تجريبية")

        # ---------------- Row 1: File input, rotation, OCR ----------------
        with gr.Row():
            file_input = gr.File(height=150, label="Choose Image or PDF (MAX: 20MB)",
                                file_types=[".pdf", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".webp"])
        # ---------------- Row 2: Rotation, OCR, Excel ----------------
        with gr.Row():
            with gr.Column(scale=1):
                rotation_dropdown = gr.Dropdown(["0", "90", "180", "270"], value="0", label="Rotate (clockwise)", interactive=False)
            with gr.Column(scale=1):
                prompt_mode = gr.Radio(["Raw text", "Text + Markdown"], value="Text + Markdown", label="OCR Output Format")
            with gr.Column(scale=1):
                ocr_btn = gr.Button("OCR", variant="primary", interactive=False)
                excel_btn = gr.DownloadButton("Export Tables to xlsx", interactive=False)

                # ---------------- Row 2: Preview and OCR Text ----------------
        with gr.Row():
            with gr.Column(scale=1):
                preview_output = gr.Image(label="Preview", type="pil", height=550)
                with gr.Row():
                    page_input = gr.Number(label="Page", value=1, precision=0)
                    total_pages_display = gr.Textbox(label="Total pages", interactive=False)

            with gr.Column(scale=1):
                text_output = gr.Textbox(label="RAW text + Markdown", lines=30)


        def show_excel_button(ocr_text):
            """Show Excel button only if OCR text contains tables"""
            if ocr_text and "<table>" in ocr_text:
                return gr.DownloadButton(interactive=True)
            else:
                return gr.DownloadButton(interactive=False)

        def show_rotation_menu(file_input):
            if file_input:
                return gr.Dropdown(interactive=True), gr.Button(interactive=True)
            else:
                return gr.Dropdown(interactive=False), gr.Button(interactive=False)

        file_input.change(
            fn=preview_loader,
            inputs=[file_input, rotation_dropdown],
            outputs=[preview_output, total_pages_display]
        ).then(
            fn=lambda total: 1,     # reset page to 1
            inputs=total_pages_display,
            outputs=page_input
        ).then(
            fn=show_rotation_menu,
            inputs=file_input,
            outputs=[rotation_dropdown, ocr_btn]
        )
        rotation_dropdown.change(
            fn=preview_loader,
            inputs=[file_input, rotation_dropdown],
            outputs=[preview_output, total_pages_display]
        ).then(
            fn=lambda total: 1,
            inputs=total_pages_display,
            outputs=page_input
        )
        page_input.change(
            fn=preview_page,
            inputs=page_input,
            outputs=preview_output
        )
        # Excel button exports tables from textbox and downloads directly
        excel_btn.click(
            fn=export_excel_from_text,
            inputs=text_output,
            outputs=excel_btn
        )
        ocr_btn.click(
            fn=run_ocr,
            inputs=[file_input, rotation_dropdown, prompt_mode],
            outputs=text_output
        ).then(
            fn=show_excel_button,
            inputs=text_output,
            outputs=excel_btn
        )
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

# --------------- Main App Loader  ----------------
if __name__ == "__main__":
    load_llm()
    launch_gradio()
