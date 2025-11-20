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
MODEL_NAME = "gemma3:4b"  
INPUT_FILE = "data_sv_NAME_30.json"
OUTPUT_FILE = "predictions_30.json"

# --- System-prompt för Modellen ---
# Denna prompt är avgörande. Den talar om för modellen exakt vad den ska göra
# och vilket format den MÅSTE svara i.
SYSTEM_PROMPT = """
Du är en expert på att extrahera information (Named Entity Recognition - NER) från svensk text.
Din uppgift är att hitta och extrahera specifika entiteter från texten som användaren tillhandahåller.

Du MÅSTE använda **exakt** följande etiketter:
- NAME: Ett separat eller komplett namn
- ADDRESS: En gatuadress, postnummer och stad.
- PHONE: Ett telefonnummer.
- EMAIL: En e-postadress.
- NATIONAL_ID: Ett svenskt personnummer (t.ex. ÅÅMMDD-XXXX, ÅÅMMDD XXXX, ÅÅMMDDXXXX).


OUTPUT-FORMAT:
Du måste svara med **enbart** en JSON-lista. Inkludera ingen förklarande text, inga ursäkter, inga kommentarer - bara JSON.
Varje objekt i listan ska ha följande exakta struktur:
{
  "label": "ETIKETT_NAMN",
  "text": "den extraherade texten",
  "start": start_index_i_texten,
  "end": end_index_i_texten
}

Om du inte hittar några entiteter alls, svara med en tom lista: []
"""

def load_data(filename):
    """Läser in JSON-datan från en fil."""
    if not os.path.exists(filename):
        print(f"FEL: Inputfilen '{filename}' hittades inte.", file=sys.stderr)
        sys.exit(1)
        
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"FEL: Kunde inte avkoda JSON från '{filename}'. Filen kan vara korrupt.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"FEL vid läsning av fil: {e}", file=sys.stderr)
        sys.exit(1)

def save_data(filename, data):
    """Sparar datan till en JSON-fil."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\nResultat sparat till '{filename}'")
    except IOError as e:
        print(f"FEL: Kunde inte skriva till filen '{filename}'. Fel: {e}", file=sys.stderr)

def get_entities_from_model(text_content):
    """Anropar Ollama-modellen för att extrahera entiteter."""
    
    # Användarens prompt är bara den rena texten, system-prompten har alla instruktioner.
    user_prompt = f"Extrahera alla entiteter från följande text och svara **endast** med den begärda JSON-listan:\n\n{text_content}"
    
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
        
        print("\n--- VARNING: Kunde inte parsa JSON-svar från modellen ---", file=sys.stderr)
    #    print(f"\n--- VARNING: Kunde inte parsa JSON-svar från modellen ---", file=sys.stderr)
        print(f"Modellens råa svar: {raw_response}", file=sys.stderr)
        return [] # Returnera en tom lista vid fel
    except Exception as e:
        # Detta fångar t.ex. anslutningsfel om 'ollama serve' inte körs
        print(f"\n--- FEL: Något gick fel vid anrop till Ollama ({e}) ---", file=sys.stderr)
        print("Kontrollera att Ollama-servern körs och att modellen '{MODEL_NAME}' är nedladdad.", file=sys.stderr)
        return None # Returnera None för att signalera ett allvarligt fel

def process_data(input_data):
    """Itererar genom indatan och bygger upp utdatan."""
    output_data = []
    
    for i, item in enumerate(input_data):
        print(f"Bearbetar nod {item.get('id', i+1)}... ", end="")
        text_to_analyze = item['text']
        
        # 1. Få entiteter från modellen
        model_entities = get_entities_from_model(text_to_analyze)
        
        if model_entities is None:
            print("Avbryter på grund av tidigare fel.")
            return None # Avbryt hela processen

        # 2. Formatera om till 'gold_entities'-struktur
        formatted_entities = []
        for j, entity in enumerate(model_entities):
            # Validera att vi har nycklarna vi behöver från modellen
            if not all(k in entity for k in ('label', 'text', 'start', 'end')):
               print(f"\n--- VARNING: Hoppar över felaktigt formaterad entitet från modell: {entity}", file=sys.stderr)
               continue

            # Extra validering: Kontrollera att modellens index stämmer
            start, end, text = entity['start'], entity['end'], entity['text']
            if text_to_analyze[start:end] != text:
                print(f"\n--- VARNING: Modellens index matchar inte text! '{text}' != '{text_to_analyze[start:end]}'. Försöker hitta texten manuellt.", file=sys.stderr)
                # Fallback: försök hitta texten manuellt
                new_start = text_to_analyze.find(text)
                if new_start != -1:
                    start = new_start
                    end = new_start + len(text)
                else:
                    print(f"--- VARNING: Kunde inte hitta '{text}' i originaltexten. Hoppar över entitet.", file=sys.stderr)
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
            "gold_entities": formatted_entities # Använder modellens prediktioner
        }
        output_data.append(new_item)
        print(f"Klar. Hittade {len(formatted_entities)} entiteter.")
        
    return output_data

# --- Huvudprogram ---
def main():
    print(f"Startar bearbetning med modell: {MODEL_NAME}")
    print(f"Läser indata från: {INPUT_FILE}")
    
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
        print("Ingen utdata genererades på grund av fel.", file=sys.stderr)

if __name__ == "__main__":
    main()