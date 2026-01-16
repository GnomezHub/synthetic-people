import os
import json
from openai import OpenAI
from flask import Flask, render_template, request, Response, stream_with_context, send_file
from werkzeug.utils import secure_filename
import io
import pdfplumber
import re
from pdf2image import convert_from_path
import pytesseract
from PIL import Image, ImageEnhance
from fpdf import FPDF

# --- CONFIGURATION ---
client = OpenAI()

MODEL_NAME = "gpt-4o" 
UPLOAD_FOLDER = 'uploads'
TEMP_FOLDER = 'temp'
ALLOWED_EXTENSIONS = {'pdf'}
LABEL_IDS = {"1", "2", "3", "4", "5"}

# Create folders if they do not exist
for folder in [UPLOAD_FOLDER, TEMP_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEMP_FOLDER'] = TEMP_FOLDER

# --- LLM SYSTEM PROMPT (Intact) ---
SYSTEM_PROMPT = """
Extract entities from the input text using ONLY the labels below.

Labels:
1 = NAME
2 = PHONE
3 = ADDRESS
4 = NATIONAL_ID
5 = EMAIL

Output one entity per line as: <label_id><entity_text>

Example text: "Anna Hansson bor på Stjärnvägen 12, Hässleholm. Hon har personnummer 950601-0909 och telefonnummer 070 091 929 3. Hennes mail är anna.hansson@live.se."

Correct output for this text is:
1Anna Hansson
3Stjärnvägen 12, Hässleholm
4950601-0909
2070 091 929 3
5anna.hansson@live.se

Rules:
- Use only these labels.
- Include every occurrence, even duplicates. - EMAIL (5) must be used for any string containing "@".
- Keep the formatting of entities exactly like it is in the original text.
- Do not output anything except the formatted lines.
- If no entities are found, output an empty string.
"""

# --- PDF EXTRACTION FUNCTIONS ---

def clean_whitespace(text: str) -> str:
    """Removes duplicated repeated whitespaces caused by OCR noise."""
    return re.sub(r" +", " ", text)

def preprocess_image(image: Image.Image) -> Image.Image:
    """Preprocesses images before OCR for better accuracy."""
    image = image.convert("L")
    image = image.point(lambda x: 0 if x < 140 else 255)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.5)
    return image

def extract_text_with_pdfplumber(pdf_path: str) -> dict:
    """Extracts text from a text-based PDF."""
    pages_text = {}
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            raw_text = page.extract_text(layout=True)
            if raw_text:
                clean_text = clean_whitespace(raw_text)
                pages_text[f"page_{i}"] = clean_text
            else:
                pages_text[f"page_{i}"] = ""
    return pages_text

def extract_text_with_ocr(pdf_path: str) -> dict:
    """Extracts text from image-based PDFs using OCR."""
    pages_text = {}
    images = convert_from_path(pdf_path, dpi=300)
    for i, image in enumerate(images, start=1):
        processed_image = preprocess_image(image)
        text = pytesseract.image_to_string(processed_image, lang="swe", config=r"--oem 3 --psm 4")
        pages_text[f"page_{i}"] = text
    return pages_text

def extract_text_from_pdf_smart(pdf_path: str) -> (dict, str):
    """Smart extractor: Falls back to OCR if text is sparse."""
    result = extract_text_with_pdfplumber(pdf_path)
    all_text = "".join(result.values()).strip()
    method = "pdfplumber (text-based)"
    if len(all_text) < 300:
        result = extract_text_with_ocr(pdf_path)
        method = "Tesseract OCR (image-based)"
    return result, method

def split_text_into_chunks_with_offsets(text: str, chunk_size: int = 500) -> list:
    """
    Splits text into chunks using direct slicing to preserve exact character indices.
    Returns a list of dictionaries with 'text' and 'start_offset'.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            # Find the last whitespace to avoid cutting a word
            last_space = text.rfind(' ', start, end)
            last_newline = text.rfind('\n', start, end)
            split_at = max(last_space, last_newline)
            if split_at > start:
                end = split_at
        
        chunks.append({
            "text": text[start:end],
            "offset": start
        })
        start = end
    return chunks

# --- PREDICTION FUNCTIONS ---

def prompt_model(text):
    """Prompts the LLM to extract PII."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Extract entities: \n\n{text}"}
            ],
            temperature=0.1,
        )
        raw_response = response.choices[0].message.content
        lines = raw_response.splitlines()
        entities = []
        for line in lines:
            line = line.strip()
            if len(line) >= 2 and line[0] in LABEL_IDS:
                entities.append((line[0], line[1:]))
        return entities, None
    except Exception as e:
        return None, str(e)

def index_finder(text, entity_texts):
    """Finds exact start/end indices in the raw chunk text."""
    found_entities = []
    unique_sorted = sorted(list(set(entity_texts)), key=len, reverse=True)
    occupied = set()
    for ent_text in unique_sorted:
        search_start = 0
        while True:
            start_idx = text.find(ent_text, search_start)
            if start_idx == -1: break
            end_idx = start_idx + len(ent_text)
            if any(i in occupied for i in range(start_idx, end_idx)):
                search_start += 1
                continue
            found_entities.append({"text": ent_text, "start": start_idx, "end": end_idx})
            for i in range(start_idx, end_idx): occupied.add(i)
            search_start = end_idx
    return found_entities

# --- FLASK ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run', methods=['POST'])
def run_script():
    file = request.files.get('file')
    if not file: return "No file", 400
    
    filename = secure_filename(file.filename)
    base_name = os.path.splitext(filename)[0]
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(pdf_path)

    def generate():
        yield f"LOG: Processing {filename}...\n"
        page_results, method = extract_text_from_pdf_smart(pdf_path)
        
        # Use a single newline to join pages to keep index math simple
        full_text = "\n".join(page_results.values())
        
        yield f"LOG: Extraction method: {method}\n"
        
        chunks = split_text_into_chunks_with_offsets(full_text)
        all_predicted = []
        label_map = {'1': 'NAME', '2': 'PHONE', '3': 'ADDRESS', '4': 'NATIONAL_ID', '5': 'EMAIL'}

        for i, chunk in enumerate(chunks):
            yield f"LOG: Analyzing chunk {i+1}/{len(chunks)}...\n"
            predictions, error = prompt_model(chunk['text'])
            if predictions:
                local_indices = index_finder(chunk['text'], [p[1] for p in predictions])
                # Match predictions to indices and apply global offset
                idx_map = {}
                for item in local_indices:
                    idx_map.setdefault(item['text'], []).append(item)
                
                for label_id, ent_text in predictions:
                    if ent_text in idx_map and idx_map[ent_text]:
                        loc = idx_map[ent_text].pop(0)
                        all_predicted.append({
                            "label": label_map.get(label_id),
                            "start": loc['start'] + chunk['offset'],
                            "end": loc['end'] + chunk['offset'],
                            "text": ent_text
                        })

        final_data = {"id": filename, "text": full_text, "predicted_entities": all_predicted}
        
        # Save JSON file with _predictions suffix
        json_path = os.path.join(app.config['TEMP_FOLDER'], f"{base_name}_predictions.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)
        
        yield f"LOG: Saved predictions to {json_path}\n"
        yield f"DATA_JSON:{json.dumps(final_data)}"

    return Response(stream_with_context(generate()), mimetype='text/plain')

@app.route('/export', methods=['POST'])
def export_pdf():
    data = request.json
    text = data.get('text', '')
    entities = data.get('entities', [])
    filename = data.get('filename', 'document.pdf')
    base_name = os.path.splitext(filename)[0]

    # Mask text in reverse to preserve indices
    entities.sort(key=lambda x: x['start'], reverse=True)
    masked_text = text
    for ent in entities:
        masked_text = masked_text[:ent['start']] + f"[{ent['label']}]" + masked_text[ent['end']:]

    # FIX UNICODE: Replace characters outside Latin-1 range to prevent FPDF crash
    clean_text = masked_text.encode("latin-1", "replace").decode("latin-1")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 5, clean_text)
    
    output = io.BytesIO()
    pdf.output(output)
    output.seek(0)
    
    return send_file(output, as_attachment=True, download_name=f"{base_name}_masked.pdf")

if __name__ == '__main__':
    app.run(debug=True)