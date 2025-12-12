import os
import json
import ollama
from flask import Flask, render_template, request, Response, stream_with_context
from werkzeug.utils import secure_filename
import io

# PDF/OCR Importer
import pdfplumber
from pathlib import Path
import re
from pdf2image import convert_from_path
import pytesseract
from PIL import Image, ImageEnhance

# --- CONFIGURATION ---
MODEL_NAME = "gemma3:4b"
UPLOAD_FOLDER = 'uploads'
TEMP_FOLDER = 'temp'
OUTPUT_FILE = "predicted-entities.json"
ALLOWED_EXTENSIONS = {'pdf'}
LABEL_IDS = {"1", "2", "3", "4", "5"}

# --- för WINDOWS (Abdo& Ernst avkommentera) ---

# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# POPPLER_PATH = r"C:\poppler-25.11.0\Library\bin" 

# Create folders if they do not exist
for folder in [UPLOAD_FOLDER, TEMP_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEMP_FOLDER'] = TEMP_FOLDER

# --- LLM SYSTEM PROMPT (From your updated get_predictions.py) ---
SYSTEM_PROMPT = """
You are extracting specific entities from text. 
You must identify ONLY the following labels, and return ONLY their numeric IDs:

1 = NAME  
2 = PHONE  
3 = ADDRESS  
4 = NATIONAL_ID
5 = EMAIL

INSTRUCTIONS:
- Read the user text.
- Find every entity that matches one of the labels above. Do not include any labels that are not present in the list above.
- For each entity, output EXACTLY ONE LINE, formatted as:

    <label_id><entity_text>

Example text: "När handläggaren på Skatteverket ringde stod det att ansökan skickats av Elin Rask. Hennes nummer är 0722 33 44 55."
The correct output for this text is:

    1Elin Rask
    20722 33 44 55

- The EMAIL label must be used for email addresses. They always contain an "@". You have to mark them as EMAIL (5), noting else. Even if they contain one or more names, they have to be marked as EMAIL.

Example: "En lärare rapporterade att e-postadressen sara.lindgren@edu.se inte gick att nå."
The correct output for this text is:

	5sara.lindgren@edu.se

- If the same entity appears more than once, ALL instances must be included. 
Example text: "Mia och Anna bor i Malmö. Anna bor mer centralt än Mia."
The correct output for this text is:

	1Mia
	1Anna
	1Anna
	1Mia

- Do NOT include commas, colons, JSON, quotes or explanations. Only the label id and entity text.
- If no entities are found, output an empty string.

Your entire response must consist ONLY of these lines, nothing else.
"""

# --- PDF EXTRACTION FUNCTIONS (From extract_pdf_text.py) ---

def fix_duplicated_text(text: str) -> str:
    """Removes duplicated repeated characters caused by OCR noise."""
    return re.sub(r"(.)\1+", r"\1", text)

def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Preprocesses images before OCR for better accuracy:
    - Converts to grayscale
    - Applies binary thresholding
    - Increases contrast
    """
    image = image.convert("L")  # Convert to grayscale
    image = image.point(lambda x: 0 if x < 140 else 255)  # Binary threshold
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.5)  # Increase contrast
    return image

def extract_text_with_pdfplumber(pdf_path: str) -> dict:
    """Extracts text from a text-based PDF using pdfplumber."""
    pages_text = {}
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            raw_text = page.extract_text(layout=True)
            if raw_text:
                clean_text = fix_duplicated_text(raw_text)
                pages_text[f"page_{i}"] = clean_text.strip()
            else:
                pages_text[f"page_{i}"] = ""
    return pages_text

def extract_text_with_ocr(pdf_path: str, poppler_path: str = None) -> dict:
    """Extracts text from image-based PDFs using OCR (Tesseract)."""
    pages_text = {}
    
    # NOTE: poppler_path must be specified if poppler is not in PATH
    images = convert_from_path(
        pdf_path,
        dpi=300,
        # poppler_path=poppler_path # UNCOMMENT IF NEEDED
    )

    for i, image in enumerate(images, start=1):
        processed_image = preprocess_image(image)
        custom_config = r"--oem 3 --psm 4"
        
        # NOTE: pytesseract.pytesseract.tesseract_cmd must be specified if tesseract
        # is not in PATH. See the top of the file.
        text = pytesseract.image_to_string(
            processed_image,
            lang="swe", # Changed to Swedish language model
            config=custom_config
        )
        pages_text[f"page_{i}"] = text.strip()
    return pages_text

def extract_text_from_pdf_smart(pdf_path: str) -> (dict, str):
    """Smart extractor: Tries text-based extraction, falls back to OCR if little text is found."""
    
    result = extract_text_with_pdfplumber(pdf_path)
    all_text = "".join(result.values()).strip()
    
    extraction_method = "pdfplumber (text-based)"

    if len(all_text) < 300:
        result = extract_text_with_ocr(pdf_path)
        extraction_method = "Tesseract OCR (image-based)"
    
    return result, extraction_method

def split_text_into_chunks(text: str, chunk_size: int = 500) -> list:
    """Splits text into chunks with a maximum size but never cuts words in half."""
    words = text.strip().split()
    chunks = []
    current_chunk = ""

    for word in words:
        if len(current_chunk) + len(word) + 1 > chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = word + " "
        else:
            current_chunk += word + " "

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

# --- PREDICTION FUNCTIONS (From get_predictions.py) ---

def prompt_model(text):
	"""Prompts the LLM and returns a list of tuples with label id and entity, and a potential error string."""
	user_prompt = f"Extract all entities from the following text and respond only with the requested entity string:\n\n{text}"
	try:
		response = ollama.chat(
			model = MODEL_NAME,
			messages=[
				{"role": "system", "content": SYSTEM_PROMPT},
				{"role": "user", "content": user_prompt}
			],
			options={"temperature": 0.1}
		)
		raw_response = response['message']['content']
	except Exception as e:
		return None, str(e)

	lines = raw_response.splitlines()
	entities = []

	for line in lines:
		line = line.strip()
		if len(line) < 2 or not line[0].isdigit(): continue
		label_id = line[0]
		if label_id not in LABEL_IDS: continue

		entities.append((label_id, line[1:])) 

	return entities, None

def index_finder(text, entity_texts):
    """
    Finds the start and end index for each entity text in the given text. 
    Handles overlapping/duplicate entities.
    """
    found_entities = []
    
    # Sort unique entity texts by length (descending) to prioritize longest entities first
    unique_sorted_texts = sorted(list(set(entity_texts)), key=len, reverse=True)
    
    # Set to keep track of occupied indices
    occupied_indices = set()

    for entity_text in unique_sorted_texts:
        search_start = 0
        while True:
            start_index = text.find(entity_text, search_start)

            if start_index == -1:
                break # No more occurrences found

            end_index = start_index + len(entity_text)
            
            # Check for index overlap
            is_occupied = any(i in occupied_indices for i in range(start_index, end_index))

            if is_occupied:
                search_start += 1 # Try searching from the next position
                continue
            else:
                found_entities.append({
                    "text": entity_text,
                    "start": start_index,
                    "end": end_index
                })
                
                # Mark positions as occupied
                for i in range(start_index, end_index):
                    occupied_indices.add(i)

                # Move search cursor forward
                search_start = end_index
    
    # Sort by start index for consistency
    found_entities.sort(key=lambda x: x['start'])
    return found_entities


def build_json(predictions, indexed_entities):
    """
    Builds the final JSON entity list format, matching up predicted entities
    with their indices, handling duplicate predictions by popping matched indices.
    """
    
    label_map = {'1': 'NAME', '2': 'PHONE', '3': 'ADDRESS', '4': 'NATIONAL_ID', '5': 'EMAIL'}
    final_entities = []
    
    # Create a lookup table where values are lists of index objects to handle duplicates
    index_lookup = {}
    for item in indexed_entities:
        index_lookup.setdefault(item["text"], []).append(item)

    for label_id, entity_text in predictions:
        index_list = index_lookup.get(entity_text)

        if index_list and len(index_list) > 0:
            # Get and remove the first index object to handle the next instance
            index_info = index_list.pop(0)
            
            label_str = label_map.get(label_id, "Unknown")

            entity_obj = {
                "label": label_str,
                "start": index_info["start"],
                "end": index_info["end"],
                "text": entity_text
            }
            final_entities.append(entity_obj)
        # else: Ignore if the model found something that is not in the text
    
    return final_entities


# --- FLASK ROUTES ---

def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run', methods=['POST'])
def run_script():
    file = request.files.get('file')
    if not file or file.filename == '':
        return "Ingen fil vald.", 400
    if not allowed_file(file.filename):
        return "Ogiltigt filformat. Endast PDF-filer stöds.", 400

    filename = secure_filename(file.filename)
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(pdf_path)

    # We use a generator function inside the route to stream the output to the client
    def generate():
        yield f"Laddade upp fil: {filename}\n"
        
        # --- STEP 1: PDF TEXT EXTRACTION ---
        yield "\n--- STEG 1: PDF TEXTEXTRAKTION ---\n"
        
        try:
            # Call the smart extractor
            page_results, method = extract_text_from_pdf_smart(pdf_path)
            
            # Concatenate all text into a single string
            full_text = "\n\n".join(page_results.values())
            
            if not full_text.strip():
                 yield "CRITICAL ERROR: Kunde inte extrahera någon text från PDF-filen.\n"
                 return

            yield f"✅ Text extraherad med: {method}. Totalt {len(full_text)} tecken."
            
            # Save the extracted text to a temporary file
            extracted_txt_path = os.path.join(app.config['TEMP_FOLDER'], "extracted.txt")
            with open(extracted_txt_path, "w", encoding="utf-8") as f:
                f.write(full_text)
            yield f" -> Fullständig text sparad till: {extracted_txt_path}"

        except Exception as e:
            # Notify the user about potential configuration issues
            yield f"CRITICAL ERROR: Fel vid PDF-extraktion. Kontrollera OCR/Poppler inställningar: {e}\n"
            return


        # --- STEP 2: TEXT CHUNKING ---
        yield "\n--- STEG 2: TEXT CHUNKNING ---\n"
        
        chunks = split_text_into_chunks(full_text, chunk_size=500)
        chunks_json_path = os.path.join(app.config['TEMP_FOLDER'], "chunks.json")
        
        with open(chunks_json_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
            
        yield f"✅ Text delad i {len(chunks)} chunks (max 500 tecken). Fil sparad: {chunks_json_path}"


        # --- STEP 3: RUN LLM PREDICTION ---
        yield "\n--- STEG 3: Get predictions  ---\n"
        
        all_predicted_entities = []
        total_chunks = len(chunks)
        # Offset to convert local chunk indices to global full_text indices
        current_char_offset = 0

        for i, chunk_text in enumerate(chunks):
            # Send status to the browser
            yield f"[{i+1}/{total_chunks}] Bearbetar chunk {i+1} ({len(chunk_text)} tecken)...\n"

            # Call the model
            predictions, error = prompt_model(chunk_text)
            
            if error:
                yield f" -> ERROR vid anrop till modell för chunk {i+1}: {error}\n"
                continue
            
            if predictions is None or len(predictions) == 0:
                yield " -> Inga entiteter hittades i denna chunk.\n"
                current_char_offset += len(chunk_text) + 2 # +2 for the separator added during join: "\n\n".join()
                continue
            
            # Extract only the text part for index finding
            entity_texts = [et for (_, et) in predictions]

            # Find local indices for the chunk
            indexed_local = index_finder(chunk_text, entity_texts)

            # Build JSON and handle duplicates
            chunk_entities = build_json(predictions, indexed_local)
            
            # Convert local indices to global indices (in full_text)
            for entity in chunk_entities:
                entity['start'] += current_char_offset
                entity['end'] += current_char_offset
                all_predicted_entities.append(entity)
            
            count = len(chunk_entities)
            yield f" -> Klart. Hittade {count} entiteter i chunk {i+1}.\n"
            
            current_char_offset += len(chunk_text) + 2 # +2 for the separator added during join: "\n\n".join()

        # --- STEP 4: FINAL COMPILATION AND STORAGE ---
        
        final_doc = {
            "id": filename, 
            "text": full_text, 
            "predicted entities": all_predicted_entities
        }

        out_path = os.path.join(app.config['TEMP_FOLDER'], OUTPUT_FILE)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([final_doc], f, ensure_ascii=False, indent=2) # [final_doc] to match array format

        yield f"\n-----------------------------------\n"
        yield f"✅ KLART! Totalt hittades {len(all_predicted_entities)} entiteter."
        yield f"Resultat sparat till: {out_path}\n"

    # Return a streaming response
    return Response(stream_with_context(generate()), mimetype='text/plain')

if __name__ == '__main__':
    print("Startar Flask server på http://127.0.0.1:5000")
    app.run(debug=True)