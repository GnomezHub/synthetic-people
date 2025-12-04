import ollama
import json
import sys
import os
import threading
from flask import Flask, render_template, request, Response, jsonify

# --- Konfiguration ---
MODEL_NAME = "gemma3:4b"  
INPUT_FILE = "data-sv-5.json"  
OUTPUT_FILE = "predictions.json"

# --- Flask-konfiguration ---
app = Flask(__name__)
# Global variabel för att hålla loggar/status i minnet.
# I en större app skulle detta vara en databas eller en kö (t.ex. Redis).
processing_log = [] 
is_processing = False

# --- System Prompt (Kopierad från originalet) ---
SYSTEM_PROMPT = """
You are an expert in extracting information (Named Entity Recognition - NER) from Swedish text.
Your task is to find and extract specific entities from the text provided by the user.

You MUST use the **exact** following labels:
- NAME: A separate or complete name
- ADDRESS: A street address, postal code, and/or city. It may be incomplete.
- PHONE: A phone number, in any common format.
- EMAIL: An email address.
- NATIONAL_ID: A Swedish personal identity number (personnummer), which may appear in various formats (e.g., YYMMDD-XXXX, YYMMDD XXXX, YYMMDDXXXX).


OUTPUT-FORMAT:
You must respond with **only** a JSON list. Include no explanatory text, no apologies, no comments - only JSON.
Each object in the list must have the following exact structure:
{
  "label": "LABEL_NAME",
  "text": "the extracted text",
  "start": start_index_in_text,
  "end": end_index_in_text
}

If you find no entities at all, respond with an empty list: []
"""
# För att göra koden kortare i detta svar, har jag kortat ner strängen ovan.
# I den faktiska filen nedan måste hela den ursprungliga SYSTEM_PROMPT klistras in.

# ----------------------------------------------------
# --- Hjälpfunktioner (anpassade för logging) ---
# ----------------------------------------------------

def log_status(message, item_id=None, level="INFO"):
    """Lägger till ett meddelande i den globala loggen."""
    global processing_log
    prefix = f"[ITEM {item_id}] " if item_id else ""
    log_entry = f"[{level}] {prefix}{message}"
    processing_log.append(log_entry)
    # Printar även till terminalen för debug
    print(log_entry, file=sys.stderr if level == "ERROR" or level == "WARNING" else sys.stdout)


def load_data(filename):
    """Läser in JSON-datan från en fil."""
    log_status(f"Läser indata från: {filename}")
    if not os.path.exists(filename):
        log_status(f"Inputfilen '{filename}' hittades inte.", level="ERROR")
        return None
        
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        log_status(f"Kunde inte avkoda JSON från '{filename}'. Filen kan vara korrupt.", level="ERROR")
        return None
    except Exception as e:
        log_status(f"FEL vid läsning av fil: {e}", level="ERROR")
        return None

def save_data(filename, data):
    """Sparar datan till en JSON-fil."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        log_status(f"Resultat sparat till '{filename}'")
    except IOError as e:
        log_status(f"Kunde inte skriva till filen '{filename}'. Fel: {e}", level="ERROR")


def get_entities_from_model(text_content):
    # ... (Ollama-anropet, nästan identiskt med originalet, men använder log_status vid fel) ...
    user_prompt = f"Extract all entities from the following text and respond **only** with the requested JSON list:\n\n{text_content}"
    
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            options={"temperature": 0.1}
        )
        
        raw_response = response['message']['content'].strip()
        
        if raw_response.startswith("```json"):
            raw_response = raw_response[7:]
        if raw_response.endswith("```"):
            raw_response = raw_response[:-3]
        
        entities = json.loads(raw_response)
        return entities

    except json.JSONDecodeError:
        log_status(f"Kunde inte parsa JSON-svar från modellen.", level="WARNING")
        # log_status(f"Modellens råa svar: {raw_response[:100]}...", level="DEBUG") # Aktivera vid djupare debug
        return []
    except Exception as e:
        log_status(f"Något gick fel vid anrop till Ollama ({e}). Kontrollera att Ollama-servern körs.", level="ERROR")
        return None

# Funktionen find_all_indices är oförändrad
def find_all_indices(parent_text, search_text):
    indices = []
    start_index = 0
    while start_index < len(parent_text):
        index = parent_text.find(search_text, start_index)
        if index == -1:
            break
        indices.append(index)
        start_index = index + len(search_text)
        if len(search_text) == 0:
            start_index += 1
    return indices

# ----------------------------------------------------
# --- Huvudbearbetningslogik (körs i separat tråd) ---
# ----------------------------------------------------

def run_ner_process():
    """Huvudfunktion för databearbetning som körs asynkront."""
    global is_processing
    global processing_log
    
    is_processing = True
    processing_log = [] # Rensa loggen vid start
    
    log_status(f"Startar bearbetning med modell: {MODEL_NAME}")
    
    # 1. Läs in indata
    input_data = load_data(INPUT_FILE)
    if not input_data:
        is_processing = False
        log_status("Bearbetning avbruten.", level="ERROR")
        return

    output_data = []
    for i, item in enumerate(input_data):
        item_id = item.get('id', i+1)
        log_status(f"Bearbetar post {i+1}/{len(input_data)}...", item_id=item_id)
        
        text_to_analyze = item['text']
        
        # 1. Få entiteter från modellen
        model_entities = get_entities_from_model(text_to_analyze)
        
        if model_entities is None:
            log_status("Avbryter bearbetning på grund av fel vid Ollama-anrop.", level="ERROR")
            is_processing = False
            return # Avbryt hela processen
            
        # 2. Hitta alla matchningar i texten för varje unik entitetstext
        all_text_matches = {} 
        for entity in model_entities:
            if 'text' in entity:
                trimmed_text = entity['text'].strip()
                if trimmed_text and trimmed_text not in all_text_matches:
                    all_text_matches[trimmed_text] = find_all_indices(text_to_analyze, trimmed_text)
        
        assigned_indices = {text: set() for text in all_text_matches.keys()}

        # 3. Korrigera entiteterna (Närmaste Index-logik)
        formatted_entities = []
        for j, entity in enumerate(model_entities):
            if not all(k in entity for k in ('label', 'text', 'start', 'end')):
               log_status(f"Hoppar över felaktigt formaterad entitet från modell: {entity}", item_id=item_id, level="WARNING")
               continue
            
            original_start = entity['start']
            entity_text = entity['text']
            trimmed_entity_text = entity_text.strip()
            
            if not trimmed_entity_text:
                log_status(f"Hoppar över entitet med tom text.", item_id=item_id, level="WARNING")
                continue
                
            match_indices = all_text_matches.get(trimmed_entity_text, [])
            
            if not match_indices:
                log_status(f"Kunde inte hitta '{trimmed_entity_text}' i texten.", item_id=item_id, level="WARNING")
                continue

            available_matches = [index for index in match_indices if index not in assigned_indices[trimmed_entity_text]]
            
            if not available_matches:
                log_status(f"För många entiteter ('{trimmed_entity_text}'). Ingen ledig position.", item_id=item_id, level="WARNING")
                continue

            # Hitta närmaste matchning
            best_new_start_index = -1
            min_distance = float('inf')
            original_start_for_dist = original_start if isinstance(original_start, int) and original_start >= 0 else 0
            
            for index in available_matches:
                distance = abs(index - original_start_for_dist)
                if distance < min_distance:
                    min_distance = distance
                    best_new_start_index = index
            
            if best_new_start_index != -1:
                new_start = best_new_start_index
                new_end = new_start + len(trimmed_entity_text)
                
                assigned_indices[trimmed_entity_text].add(new_start)

                formatted_entities.append({
                    "id": f"e{j+1}", 
                    "label": entity['label'],
                    "start": new_start,
                    "end": new_end,
                    "text": trimmed_entity_text
                })
        
        # 4. Bygg det nya objektet
        new_item = {
            "id": item_id,
            "language": item.get('language'),
            "text": item['text'],
            "predicted_entities": formatted_entities
        }
        output_data.append(new_item)
        log_status(f"Klar. Hittade {len(formatted_entities)} entiteter.", item_id=item_id)
        
    # 5. Spara utdata
    save_data(OUTPUT_FILE, output_data)
    log_status("Bearbetning avslutad. Resultat sparat.", level="SUCCESS")
    is_processing = False

# ----------------------------------------------------
# --- Flask Routes ---
# ----------------------------------------------------

@app.route("/")
def index():
    """Huvudsida med knappen och statusfältet."""
    return render_template("index.html")

@app.route("/start_process", methods=["POST"])
def start_process():
    """Startar bearbetningen i en separat tråd."""
    global is_processing
    if is_processing:
        return jsonify({"status": "error", "message": "Bearbetning körs redan."}), 409
    
    # Starta bearbetningen i en bakgrundstråd
    thread = threading.Thread(target=run_ner_process)
    thread.start()
    
    return jsonify({"status": "success", "message": "Bearbetning startad i bakgrunden."})

@app.route("/status_stream")
def status_stream():
    """Server-Sent Events (SSE) route för realtidsuppdateringar."""
    
    def event_stream():
        """Generator som skickar nya loggmeddelanden till klienten."""
        global processing_log
        current_index = 0
        
        while is_processing or current_index < len(processing_log):
            if current_index < len(processing_log):
                # Hämta nya meddelanden
                new_messages = processing_log[current_index:]
                current_index = len(processing_log)
                
                # Skicka varje meddelande som ett SSE
                for msg in new_messages:
                    yield f"data: {json.dumps({'log': msg})}\n\n"
            
            elif not is_processing and current_index == len(processing_log):
                # Skicka ett avslutningsmeddelande när bearbetningen är klar och alla loggar skickats
                yield "data: {\"log\": \"__END__\"}\n\n"
                break
                
            # Vänta en kort stund innan vi kontrollerar loggen igen
            threading.Event().wait(0.5) 

    # Returnera en Flask Response med rätt MIME-typ för SSE
    return Response(event_stream(), mimetype="text/event-stream")

@app.route("/is_running")
def is_running():
    """API-slutpunkt för att kontrollera om processen körs."""
    return jsonify({"is_running": is_processing})


if __name__ == "__main__":
    # Flask-appen kommer att hantera synkronisering till log_status via den globala variabeln.
    # För en produktionsapp rekommenderas starkt en riktig kölösning (Celery/Redis) istället.
    app.run(debug=True, threaded=True) # Sätt threaded=True för att tillåta flera anslutningar