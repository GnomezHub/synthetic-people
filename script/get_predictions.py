"""
Läser in json filen angiven i variabeln INPUT_FILE. Använder modellen angiven i MODEL_NAME
med instruktionerna i SYSTEM_PROMPT för varje textsnutt i json filen. 
Ett output objekt skapas med all innehåll kopierad från input json filen, förutom entiteterna som fås av modellen
Entiteterna matchas med texten för att se till att indexerna är rätt, är de inte det korrigeras de.
(indexerna från modellen används för att korrigera rätt ord vid flera förekomster)
resultatet sparas under filnamnet som anges av OUTPUT_FILE

"""
#RÄTT INDEX-ÅTGÄRD - UPPDATERAD MED "NÄRMASTE INDEX"-LOGIK

import ollama
import json
import sys
import os

# --- Konfiguration ---
MODEL_NAME = "gemma3:4b"  
INPUT_FILE = "data-sv-30.json"  #
OUTPUT_FILE = "predictions.json"

# --- System Prompt for the Model ---
# Systemprompten med instruktionerna

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
    user_prompt = f"Extract all entities from the following text and respond **only** with the requested JSON list:\n\n{text_content}"
    
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            options={
                "temperature": 0.1  # Låg temperatur för mer konsekventa JSON-svar
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
        print(f"Modellens råa svar: {raw_response}", file=sys.stderr)
        return [] # Returnera en tom lista vid fel
    except Exception as e:
        # Detta fångar t.ex. anslutningsfel om 'ollama serve' inte körs
        print(f"\n--- FEL: Något gick fel vid anrop till Ollama ({e}) ---", file=sys.stderr)
        print(f"Kontrollera att Ollama-servern körs och att modellen '{MODEL_NAME}' är nedladdad.", file=sys.stderr)
        return None # Returnera None för att signalera ett allvarligt fel

def find_all_indices(parent_text, search_text):
    """Hittar alla startindex för en given textsträng i en text."""
    indices = []
    start_index = 0
    while start_index < len(parent_text):
        index = parent_text.find(search_text, start_index)
        if index == -1:
            break
        indices.append(index)
        # Hoppa framåt för att hitta nästa förekomst
        start_index = index + len(search_text)
        # Säkerhetsstopp
        if len(search_text) == 0:
            start_index += 1
    return indices

def process_data(input_data):
    """Itererar genom indatan och bygger upp utdatan, korrigerar index med närmaste-matchningslogik."""
    output_data = []
    
    for i, item in enumerate(input_data):
        print(f"Bearbetar nod {item.get('id', i+1)}... ", end="")
        text_to_analyze = item['text']
        
        # 1. Få entiteter från modellen
        model_entities = get_entities_from_model(text_to_analyze)
        
        if model_entities is None:
            print("Avbryter på grund av tidigare fel.")
            return None # Avbryt hela processen

        # 2. Förbered för indexkorrigering
        
        # Steg A: Hitta alla matchningar i texten för varje unik entitetstext
        # Vi använder en dictionary för att lagra alla hittade startpositioner
        all_text_matches = {} 
        for entity in model_entities:
            # Kontrollera att vi har en 'text'-nyckel och trimma den
            if 'text' in entity:
                trimmed_text = entity['text'].strip()
                if trimmed_text and trimmed_text not in all_text_matches:
                    all_text_matches[trimmed_text] = find_all_indices(text_to_analyze, trimmed_text)
        
        # Steg B: Håll reda på vilka startindex som har använts (tilldelats)
        assigned_indices = {text: set() for text in all_text_matches.keys()}


        # 3. Korrigera entiteterna
        formatted_entities = []
        for j, entity in enumerate(model_entities):
            
            # Validera att vi har de nycklar vi behöver från modellen
            if not all(k in entity for k in ('label', 'text', 'start', 'end')):
               print(f"\n--- VARNING: Hoppar över felaktigt formaterad entitet från modell: {entity}", file=sys.stderr)
               continue
            
            # Hämta värden, trimma texten för sökning
            original_start = entity['start']
            entity_text = entity['text']
            trimmed_entity_text = entity_text.strip()
            
            if not trimmed_entity_text:
                print(f"\n--- VARNING: Hoppar över entitet med tom text: {entity}", file=sys.stderr)
                continue
                
            # Hämta alla möjliga matchningar för denna textsträng
            match_indices = all_text_matches.get(trimmed_entity_text, [])
            
            if not match_indices:
                print(f"\n--- VARNING: Kunde inte hitta '{trimmed_entity_text}' i texten. Hoppar över entitet.", file=sys.stderr)
                continue # Texten hittades inte alls

            # Hitta tillgängliga matchningar som inte redan har tilldelats
            available_matches = [
                index for index in match_indices 
                if index not in assigned_indices[trimmed_entity_text]
            ]
            
            if not available_matches:
                # Detta kan hända om det finns fler entiteter än matchningar (tvetydigt)
                print(f"\n--- VARNING: För många entiteter ('{trimmed_entity_text}'). Ingen ledig position hittades. Hoppar över.", file=sys.stderr)
                continue

            # Hitta den tillgängliga matchningen som ligger närmast det ursprungliga indexet
            best_new_start_index = -1
            min_distance = float('inf')
            
            # Använd 0 som fallback om original_start inte är ett giltigt numeriskt index
            original_start_for_dist = original_start if isinstance(original_start, int) and original_start >= 0 else 0
            
            # 1. Sök efter närmaste index
            for index in available_matches:
                distance = abs(index - original_start_for_dist)
                if distance < min_distance:
                    min_distance = distance
                    best_new_start_index = index
            
            # 2. Tilldela det funna indexet
            if best_new_start_index != -1:
                new_start = best_new_start_index
                new_end = new_start + len(trimmed_entity_text)
                
                # Registrera indexet som använt
                assigned_indices[trimmed_entity_text].add(new_start)

                # Lägg till den korrigerade entiteten
                formatted_entities.append({
                    "id": f"e{j+1}", 
                    "label": entity['label'],
                    "start": new_start,
                    "end": new_end,
                    "text": trimmed_entity_text # Använd den trimmade texten för konsekvens
                })
            else:
                # Logikfel/bör inte hända om available_matches inte var tom
                print(f"\n--- FEL: Internt logikfel vid tilldelning för '{trimmed_entity_text}'. Hoppar över.", file=sys.stderr)
        
        # 4. Bygg det nya objektet för output-filen
        new_item = {
            "id": item['id'],
            "language": item['language'],
            "text": item['text'],
            "predicted_entities": formatted_entities # Använder modellens korrigerade prediktioner
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