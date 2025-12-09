import os
import json
import ollama
from flask import Flask, render_template, request, Response, stream_with_context
from werkzeug.utils import secure_filename


MODEL_NAME = "gemma3:4b"
UPLOAD_FOLDER = 'uploads'
OUTPUT_FILE = "predictions-output.json"
LABEL_IDS = {"1", "2", "3", "4", "5"}

# create uppload folder if necesärry
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


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

- Do NOT include commas, colons, JSON, quotes or explanations. Only the label id and entity text.
- If no entities are found, output an empty string.

Your entire response must consist ONLY of these lines, nothing else.
"""



def prompt_model(text):
    """SPrompts the LLM for one document and returns a list of tuples with label id and entity. Med felhantering som inte dödar servern."""
    user_prompt = f"Extract all entities from the following text and respond only with the requested entity string:\n\n{text}"
    try:
        response = ollama.chat(
            model=MODEL_NAME,
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
    seen = set()
    entities = []

    for line in lines:
        line = line.strip()
        if len(line) < 2 or not line[0].isdigit(): continue
        label_id = line[0]
        if label_id not in LABEL_IDS or line in seen: continue

        seen.add(line)
        entities.append((label_id, line[1:]))

    return entities, None

def index_finder(text, entity_texts):

    found_entities = []
    occupied_indices = set()
    unique_entities = sorted(list(set(entity_texts)), key=len, reverse=True)

    for entity_text in unique_entities:
        search_start = 0
        while True:
            start_index = text.find(entity_text, search_start)
            if start_index == -1: break
            
            end_index = start_index + len(entity_text)
            is_occupied = any(i in occupied_indices for i in range(start_index, end_index))

            if is_occupied:
                search_start += 1
                continue
            else:
                found_entities.append({"text": entity_text, "start": start_index, "end": end_index})
                for i in range(start_index, end_index): occupied_indices.add(i)
                search_start = end_index
    
    found_entities.sort(key=lambda x: x['start'])
    return found_entities

def build_json(predictions, indexed_entities):
	"""
    Takes:
        predictions = [(label_id, entity_text), ...]
        indexed_entities = [ {"text": ..., "start": ..., "end": ...}, ...]

    Returns:
        A list of JSON dictionaries matching the gold format:
		[{'label': 'NAME', 'start': 23, 'end': 27, 'text': 'Elin Rask'}, {'label': 'PHONE', 'start': 12, 'end': 48, 'text': '0722 33 44 55'}]
    """

	label_map = {
        '1': 'NAME',
        '2': 'PHONE',
        '3': 'ADDRESS',
        '4': 'NATIONAL_ID',
		'5': 'EMAIL'
    }

	final_entities = []
 
	index_lookup = {}
	for item in indexed_entities:
		text = item["text"]
		index_lookup.setdefault(text, []).append(item)

 
	for label_id, entity_text in predictions:
		index_list = index_lookup.get(entity_text)

		if not index_list or len(index_list) == 0:
			print(f"WARNING: No remaining index match for entity '{entity_text}'")
			continue

		index_info = index_list.pop(0)

		# Convert label_id -> label string
		label_str = label_map.get(label_id, "Unknown")

		entity_obj = {
			"label": label_str,
			"start": index_info["start"],
			"end": index_info["end"],
			"text": entity_text
		}

		final_entities.append(entity_obj)
	
	return final_entities

# --- FLASK ROUTER ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run', methods=['POST'])
def run_script():
    file = request.files.get('file')
    if not file:
        return "Ingen fil vald.", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # use generete to streama output
    def generate():
        yield f"Laddade upp fil: {filename}\n"
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                docs = json.load(f)
        except Exception as e:
            yield f"CRITICAL ERROR: Kunde inte läsa JSON: {e}\n"
            return

        output_docs = []
        total_docs = len(docs)

        for i, doc in enumerate(docs):
            doc_id = doc.get("id", "Unknown")
            text = doc.get("text", "")
            
            # Send to browsööör
            yield f"[{i+1}/{total_docs}] Bearbetar dokument ID: {doc_id}...\n"

            predictions, error = prompt_model(text)
            
            if error:
                yield f" -> ERROR vid anrop till modell: {error}\n"
                continue
            
            if predictions is None:
                yield " -> Inga predictions eller fel vid parsing.\n"
                continue
            
            entity_texts = [et for (_, et) in predictions]
            indexed = index_finder(text, entity_texts)
            predicted_entities = build_json(predictions, indexed)

            output_doc = {
                "id": doc_id, 
                "language": doc.get("language", ""),
                "text": text, 
                "predicted entities": predicted_entities
            }
            output_docs.append(output_doc)
            
            # Show me the moneeeey
            count = len(predicted_entities)
            yield f" -> Klart. Hittade {count} entiteter.\n"

        # Savit
        out_path = os.path.join(app.config['UPLOAD_FOLDER'], OUTPUT_FILE)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_docs, f, ensure_ascii=False, indent=2)
        
        yield "\n-----------------------------------\n"
        yield f"KLART! Resultat sparat till: {out_path}\n"

    # return that bitch
    return Response(stream_with_context(generate()), mimetype='text/plain')

if __name__ == '__main__':
    print("Startar server på http://127.0.0.1:5000")
    app.run(debug=True)