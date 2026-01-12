import os
import json
import io
import re
from flask import Flask, render_template, request, Response, stream_with_context, send_file
from werkzeug.utils import secure_filename
from openai import OpenAI
import pdfplumber
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

for folder in [UPLOAD_FOLDER, TEMP_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEMP_FOLDER'] = TEMP_FOLDER

# --- LLM SYSTEM PROMPT (Intact as requested) ---
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

# --- PDF & TEXT UTILS ---

def extract_text_smart(pdf_path):
    """Extracts text using pdfplumber, falls back to OCR if needed."""
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t: full_text += t + "\n"
    
    if len(full_text.strip()) < 200:
        images = convert_from_path(pdf_path, dpi=300)
        full_text = ""
        for img in images:
            img = img.convert("L").point(lambda x: 0 if x < 140 else 255)
            full_text += pytesseract.image_to_string(img, lang="swe") + "\n"
    return full_text

def get_chunks(text, size):
    """Splits text into chunks with global offsets to preserve indices."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        if end < len(text):
            last_space = text.rfind(' ', start, end)
            if last_space > start: end = last_space
        chunks.append({"text": text[start:end], "offset": start})
        start = end
    return chunks

# --- PREDICTION LOGIC ---

def prompt_llm(text):
    """Calls OpenAI to predict PII entities."""
    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, 
                      {"role": "user", "content": f"Extract entities:\n\n{text}"}],
            temperature=0.1
        )
        lines = res.choices[0].message.content.splitlines()
        return [ (l[0], l[1:]) for l in lines if len(l) > 1 and l[0] in LABEL_IDS ]
    except: return []

def find_indices(text, entities):
    """Finds exact start/end in text for found entities."""
    results = []
    occupied = set()
    sorted_ents = sorted(list(set([e[1] for e in entities])), key=len, reverse=True)
    
    label_map = {'1':'NAME','2':'PHONE','3':'ADDRESS','4':'NATIONAL_ID','5':'EMAIL'}
    ent_labels = {e[1]: label_map.get(e[0]) for e in entities}

    for val in sorted_ents:
        start_search = 0
        while True:
            idx = text.find(val, start_search)
            if idx == -1: break
            end = idx + len(val)
            if not any(i in occupied for i in range(idx, end)):
                results.append({"text": val, "start": idx, "end": end, "label": ent_labels[val]})
                for i in range(idx, end): occupied.add(i)
            start_search = idx + 1
    return results

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run', methods=['POST'])
def run_process():
    file = request.files.get('file')
    chunk_size = int(request.form.get('chunk_size', 1000))
    if not file: return "No file", 400
    
    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)

    def generate():
        yield "LOG: Extracting text...\n"
        text = extract_text_smart(path)
        chunks = get_chunks(text, chunk_size)
        yield f"LOG: Created {len(chunks)} chunks.\n"
        
        all_results = []
        for i, c in enumerate(chunks):
            yield f"LOG: Processing chunk {i+1}/{len(chunks)}...\n"
            preds = prompt_llm(c['text'])
            indexed = find_indices(c['text'], preds)
            # Add global offset
            for item in indexed:
                item['start'] += c['offset']
                item['end'] += c['offset']
            all_results.extend(indexed)
        
        final_doc = {"id": filename, "text": text, "entities": all_results}
        
        # Save JSON predictions
        json_name = os.path.splitext(filename)[0] + "_predictions.json"
        with open(os.path.join(app.config['TEMP_FOLDER'], json_name), "w", encoding="utf-8") as f:
            json.dump(final_doc, f, ensure_ascii=False, indent=2)
            
        yield f"LOG: JSON saved as {json_name}\n"
        yield f"DATA_JSON:{json.dumps(final_doc)}"

    return Response(stream_with_context(generate()), mimetype='text/plain')

@app.route('/export', methods=['POST'])
def export():
    data = request.json
    text = data['text']
    entities = data['entities']
    m_type = data['mask_type']
    filename = data['filename']
    
    # Sort reverse to not break indices
    entities.sort(key=lambda x: x['start'], reverse=True)
    
    for ent in entities:
        val = ""
        if m_type == "label": val = f"[{ent['label']}]"
        elif m_type == "xxx": val = "X" * (ent['end'] - ent['start'])
        elif m_type == "stars": val = "*" * (ent['end'] - ent['start'])
        else: val = "█" * (ent['end'] - ent['start']) # Redacted style
        
        text = text[:ent['start']] + val + text[ent['end']:]

    # FPDF Setup with Unicode support
    pdf = FPDF()
    pdf.add_page()
    # Note: To support Swedish characters, ensure DejaVuSans.ttf is in the folder
    # If not present, it will fallback to standard Latin-1 which might fail on '█'
    try:
        # If you have the font file, uncomment these:
        # pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
        # pdf.set_font('DejaVu', size=10)
        pdf.set_font("Courier", size=10) # Courier has better block character support in some viewers
    except:
        pdf.set_font("Arial", size=10)

    # Clean text for Latin-1 if no Unicode font is found
    safe_text = text.encode('utf-8', 'replace').decode('utf-8')
    pdf.multi_cell(0, 5, safe_text)
    
    out = io.BytesIO()
    pdf.output(out)
    out.seek(0)
    
    new_name = os.path.splitext(filename)[0] + "_masked.pdf"
    return send_file(out, as_attachment=True, download_name=new_name)

if __name__ == '__main__':
    app.run(debug=True)