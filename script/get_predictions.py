#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 12:41:48 2025

@author: Danny Gomez
"""

import ollama
import json
import sys
import os

# --- Konfiguration ---
MODEL_NAME = "gemma3:4b"  # Se till att du har denna modell: `ollama pull gemma3:4b`
INPUT_FILE = "data_short.json"
OUTPUT_FILE = "data_short_out.json"

# --- System-prompt för Modellen ---
# Denna prompt är avgörande. Den talar om för modellen exakt vad den ska göra
# och vilket format den MÅSTE svara i.
SYSTEM_PROMPT = """
You are a strict, rule-based Named Entity Recognition (NER) model. 
Your task is to extract personal data entities from Swedish text. You will receive 
a text from the user and must identify all occurrences of the following entity types:

REQUIRED LABELS:
- FIRST_NAME: A standalone first name.
- LAST_NAME: A standalone last name.
- FULL_NAME: A complete name where first name and last name (and optional middle names) 
             appear directly next to each other in the text.
- ADDRESS: A street address, postal code, or city (e.g., “Storgatan 12, 123 45 Malmö”).
- PHONE: Any Swedish-format phone number.
- EMAIL: An email address.
- NATIONAL_ID: A Swedish personal identity number (personnummer), in formats 
               YYMMDD-XXXX, YYMMDDXXXX, YYYYMMDDXXXX or YYYYMMDD-XXXX.

RULES FOR NAMES:
1. If a first name and a last name appear directly next to each other, 
   you MUST use FULL_NAME only (not FIRST_NAME + LAST_NAME).
2. Use FIRST_NAME and LAST_NAME only when they appear separately.
3. FULL_NAME may contain middle names.

INDEXING RULES:
- Use Python-style indexing:
  * start = index of the first character (0-based)
  * end = index of the first character after the entity
- Indices MUST match the text exactly.
- The “text” field must contain the exact substring from the input 
  (preserving casing, spaces, accents, and hyphens).

OUTPUT FORMAT (VERY IMPORTANT):
You must respond with a JSON array only. No explanations, no comments, no markdown fences.
Each entity must be an object of the form:

{
  "label": "LABEL_NAME",
  "text": "exact entity text",
  "start": integer,
  "end": integer
}

If no entities are found, return an empty list: []
"""

def load_data(filename):
    """Läser in JSON-datan från en fil."""
    if not os.path.exists(filename):
        print(f"ERROR: Input file '{filename}' wasn't found.", file=sys.stderr)
        sys.exit(1)
        
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"ERROR: Couldn't decode JSON from '{filename}'. The file might be corrupted.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR reading file: {e}", file=sys.stderr)
        sys.exit(1)

def save_data(filename, data):
    """Sparar datan till en JSON-fil."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\nResult saved to '{filename}'")
    except IOError as e:
        print(f"ERROR: Couldn't write data to file'{filename}'. Error: {e}", file=sys.stderr)

def get_entities_from_model(text_content):
    """Anropar Ollama-modellen för att extrahera entiteter."""
    
    # Användarens prompt är bara den rena texten, system-prompten har alla instruktioner.
    user_prompt = f"Extract all entities from the following text and reply ONLY with the requested JSON-list: \n\n{text_content}"
    
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            options={
                "temperature": 0.0  # Låg temperatur för mer konsekventa JSON-svar
            }
        )
        
        raw_response = response['message']['content'].strip()
        
        # Försök att rensa bort eventuella "markdown" ```json ... ``` som modellen kan lägga till
        if raw_response.startswith("```json"):
            raw_response = raw_response[7:]
        if raw_response.endswith("```"):
            raw_response = raw_response[:-3]
        
        # Försök att parsa JSON
        entities = json.loads(raw_response)
        return entities

    except json.JSONDecodeError:
        
        print("\n--- WARNING: Couldn't parse JSON response from model ---", file=sys.stderr)
    #    print(f"\n--- VARNING: Kunde inte parsa JSON-svar från modellen ---", file=sys.stderr)
        print(f"Model's raw response: {raw_response}", file=sys.stderr)
        return [] # Returnera en tom lista vid fel
    except Exception as e:
        # Detta fångar t.ex. anslutningsfel om 'ollama serve' inte körs
        print(f"\n--- ERROR: Something went wrong when calling Ollama ({e}) ---", file=sys.stderr)
        print("Check that the Ollama server is running and that the model '{MODEL_NAME}' is downloaded.", file=sys.stderr)
        return None # Returnera None för att signalera ett allvarligt fel

def process_data(input_data):
    """Itererar genom indatan och bygger upp utdatan."""
    output_data = []
    
    for i, item in enumerate(input_data):
        print(f"Processing node {item.get('id', i+1)}... ", end="")
        text_to_analyze = item['text']
        
        # 1. Få entiteter från modellen
        model_entities = get_entities_from_model(text_to_analyze)
        
        if model_entities is None:
            print("Interrupting process.")
            return None # Avbryt hela processen

        # 2. Formatera om till 'gold_entities'-struktur
        formatted_entities = []
        for j, entity in enumerate(model_entities):
            # Validera att vi har nycklarna vi behöver från modellen
            if not all(k in entity for k in ('label', 'text', 'start', 'end')):
               print(f"\n--- WARNING: Skipping wrongly formatted entity from model: {entity}", file=sys.stderr)
               continue

            # Extra validering: Kontrollera att modellens index stämmer
            start, end, text = entity['start'], entity['end'], entity['text']
            if text_to_analyze[start:end] != text:
                print(f"\n--- WARNING: Model index doesn't match text: '{text}' != '{text_to_analyze[start:end]}'. Trying to find text manually.", file=sys.stderr)
                # Fallback: försök hitta texten manuellt
                new_start = text_to_analyze.find(text)
                if new_start != -1:
                    start = new_start
                    end = new_start + len(text)
                else:
                    print(f"--- WARNING: Couldn't find '{text}' in the original text. Skipping entity.", file=sys.stderr)
                    continue # Hoppa över denna felaktiga entitet

            formatted_entities.append({
                "id": f"e{j+1}", # Skapa nytt ID (e1, e2, ...)
                "label": entity['label'],
                "start": start,
                "end": end,
                "text": text
            })
        
        # 3. Bygg det nya objektet för output-filen
        new_item = {
            "id": item['id'],
            "language": item['language'],
            "text": item['text'],
            "predicted_entities": formatted_entities # Använder modellens prediktioner
        }
        output_data.append(new_item)
        print(f"Done. Found {len(formatted_entities)} entities.")
        
    return output_data

# --- Huvudprogram ---
def main():
    print(f"Starting to process with model: {MODEL_NAME}")
    print(f"Reading input data from: {INPUT_FILE}")
    
    # 1. Läs in indata
    input_data = load_data(INPUT_FILE)
    if not input_data:
        return # Avsluta om indata inte kunde läsas

    # 2. Bearbeta datan
    predictions = process_data(input_data)
    
    # 3. Spara utdata
    if predictions:
        save_data(OUTPUT_FILE, predictions)
    else:
        print("No output data was generated due to error.", file=sys.stderr)

if __name__ == "__main__":
    main()